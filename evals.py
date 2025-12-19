import numpy as np
import porespy as ps
import openpnm as op
import scipy.ndimage as spim
from skimage import io
from skimage import filters

import scipy.sparse.csgraph as csgraph
from scipy import sparse

import matplotlib.pyplot as plt
from scipy import stats


def evaluate_structure(im, voxel_size=3e-9):
    """
    Evaluates morphological and transport properties of a 3D binary structure.
    """
    im = im.astype(bool)
    
    # --- 1. Porosity ---
    porosity_total = im.sum() / im.size
    labels = spim.label(im)[0]
    largest_cluster = labels == np.argmax(np.bincount(labels[labels > 0]))
    porosity_connected = largest_cluster.sum() / im.size

    # --- 2. Pore Network Extraction ---
    print("  > Extracting pore network (this may take a moment)...")
    net_ext = ps.networks.snow2(im, voxel_size=voxel_size, boundary_width=0)
    
    try:
        net = op.io.network_from_porespy(net_ext.network)
    except AttributeError:
        net = op.network.Network()
        net.update(net_ext.network)

    # --- CRITICAL FIX START: Add Geometry Models ---
    # 1. Alias the diameters so OpenPNM models can find them
    net['pore.diameter'] = net['pore.inscribed_diameter']
    net['throat.diameter'] = net['throat.inscribed_diameter']
    
    # 2. Add a geometry collection (Spheres & Cylinders)
    # This provides the equations to calculate 'size_factors' from diameters
    net.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    
    # 3. Regenerate models to calculate the new geometry props
    net.regenerate_models()
    # --- CRITICAL FIX END ---

    # Get pore & throat geometry (for stats)
    pore_radii = net['pore.diameter'] / 2
    throat_radii = net['throat.diameter'] / 2
    conns = net['throat.conns']
    coordination_num = np.bincount(conns.flatten(), minlength=net.Np)

    # --- 3. Tortuosity (Geometric via Graph) ---
    print("  > Calculating tortuosity (via Network Graph)...")
    L = im.shape[2] * voxel_size
    z_coords = net['pore.coords'][:, 2]
    
    # Identify Inlets/Outlets
    inlet_pores = np.where(z_coords <= z_coords.min() + voxel_size)[0]
    outlet_pores = np.where(z_coords >= z_coords.max() - voxel_size)[0]
    
    if len(inlet_pores) > 0 and len(outlet_pores) > 0:
        P1 = conns[:, 0]
        P2 = conns[:, 1]
        C1 = net['pore.coords'][P1]
        C2 = net['pore.coords'][P2]
        throat_lengths = np.linalg.norm(C1 - C2, axis=1)
        
        graph = sparse.coo_matrix((throat_lengths, (P1, P2)), shape=(net.Np, net.Np))
        graph = graph + graph.T
        
        dist_matrix = csgraph.dijkstra(graph, indices=inlet_pores)
        dists_to_outlets = dist_matrix[:, outlet_pores]
        min_dists = np.min(dists_to_outlets, axis=0)
        valid_paths = min_dists[~np.isinf(min_dists)]
        
        if len(valid_paths) > 0:
            tortuosity = np.mean(valid_paths) / L
        else:
            tortuosity = 1.0 
    else:
        tortuosity = 1.0

    # --- 4. Permeability (Stokes Flow Simulation) ---
    print("  > Simulating permeability...")
    water = op.phase.Water(network=net)
    
    # Add physics models (now that geometry is fixed, this will work)
    water.add_model_collection(op.models.collections.physics.standard)
    water.regenerate_models()
    
    flow = op.algorithms.StokesFlow(network=net, phase=water)
    
    # Apply BCs
    if len(inlet_pores) == 0:
        inlet_pores = np.where(z_coords <= z_coords.min() + voxel_size)[0]
    if len(outlet_pores) == 0:
        outlet_pores = np.where(z_coords >= z_coords.max() - voxel_size)[0]

    flow.set_value_BC(pores=inlet_pores, values=210000) 
    flow.set_value_BC(pores=outlet_pores, values=0)
    flow.run()
    
    Q = flow.rate(pores=outlet_pores)[0] 
    A = im.shape[0] * im.shape[1] * (voxel_size**2)
    
    # Use fluid viscosity
    mu = water['pore.viscosity'].mean() 
    dP = 210000 
    
    flux_m_s = Q / A
    flux_LMH = flux_m_s * 1000 * 3600
    dP_psi = dP * 0.000145038
    
    if dP_psi > 0:
        perm_LMH_psi = flux_LMH / dP_psi
    else:
        perm_LMH_psi = 0
    
    pore_radii_nm = (net['pore.diameter'] / 2) * 1e9
    throat_radii_nm = (net['throat.diameter'] / 2) * 1e9

    return {
        "mean_pore_radius_nm": np.mean(pore_radii) * 1e9,
        "mean_throat_radius_nm": np.mean(throat_radii) * 1e9,
        "mean_coord_num": np.mean(coordination_num),
        "porosity_total": porosity_total,
        "porosity_connected": porosity_connected,
        "tortuosity": tortuosity,
        "permeability_LMH_psi": perm_LMH_psi,
        "dist_pore_radii": pore_radii_nm,
        "dist_throat_radii": throat_radii_nm,
        "dist_coord_num": coordination_num
        
    }



def calculate_statistics(results_list):
    """
    Calculates Mean, SD, 95% CI for a list of result dictionaries.
    """
    keys = results_list[0].keys()
    stats_summary = {}
    
    for k in keys:
        values = [r[k] for r in results_list]
        mean = np.mean(values)
        sd = np.std(values, ddof=1) # Sample standard deviation
        n = len(values)
        
        # 95% CI Margin of Error formula from paper: 1.96 * (SD / sqrt(N))
        margin_error = 1.96 * (sd / np.sqrt(n))
        
        stats_summary[k] = {
            "mean": mean,
            "std": sd,
            "95_CI": margin_error
        }
        
    return stats_summary

def print_table_comparison(original_stats, reconstruction_stats):
    """Prints the comparison table similar to Table 1"""
    
    headers = ["Property", "Original", "Reconstructed (Mean ± 95% CI)", "Error (%)"]
    row_fmt = "{:<25} {:<15} {:<30} {:<10}"
    
    print("\n" + "="*80)
    print(row_fmt.format(*headers))
    print("-" * 80)
    
    mapping = [
        ("Mean Body Radius (nm)", "mean_pore_radius_nm"),
        ("Mean Throat Radius (nm)", "mean_throat_radius_nm"),
        ("Coordination Number", "mean_coord_num"),
        ("Porosity (Total)", "porosity_total"),
        ("Porosity (Connected)", "porosity_connected"),
        ("Tortuosity", "tortuosity"),
        ("Permeability (LMH/psi)", "permeability_LMH_psi")
    ]
    
    for label, key in mapping:
        orig_val = original_stats[key]
        recon_mean = reconstruction_stats[key]['mean']
        recon_ci = reconstruction_stats[key]['95_CI']
        
        # Calculate Error %: |Original - Recon| / Original * 100
        error_pct = abs(orig_val - recon_mean) / orig_val * 100
        
        recon_str = f"{recon_mean:.2f} ± {recon_ci:.2f}"
        print(row_fmt.format(label, f"{orig_val:.2f}", recon_str, f"{error_pct:.1f}%"))
    print("="*80 + "\n")

import matplotlib.pyplot as plt
from scipy import stats

def get_histogram_stats(data_list, bins, density=False):
    """
    Computes mean frequency and 95% CI per bin across N reconstructions.
    """
    n_recons = len(data_list)
    hist_matrix = []
    
    for data in data_list:
        # Calculate histogram for this specific reconstruction
        # weights argument ensures 'frequency' is a percentage if needed
        hist, edges = np.histogram(data, bins=bins, density=density)
        
        # Convert to percentage frequency if density=False
        if not density:
            hist = (hist / len(data)) * 100
            
        hist_matrix.append(hist)
    
    hist_matrix = np.array(hist_matrix) # Shape: (N_recons, N_bins)
    
    # Calculate Stats per bin
    mean_hist = np.mean(hist_matrix, axis=0)
    std_hist = np.std(hist_matrix, axis=0, ddof=1)
    
    # 95% CI = 1.96 * SE
    ci_hist = 1.96 * (std_hist / np.sqrt(n_recons))
    
    return mean_hist, ci_hist, edges

def generate_figure_3(original_res, recon_results_list):
    """
    Generates Figure 3 B, C, D comparing Original vs Reconstructed (Mean +/- 95% CI)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define Plot Configs
    # (Data Key, Bin Config, X Label, Title, Axis Index)
    configs = [
        ("dist_pore_radii", np.linspace(0, 150, 16), "Pore radius (nm)", "B: Pore Radius Distribution", 0),
        ("dist_throat_radii", np.linspace(0, 70, 15), "Throat radius (nm)", "C: Throat Radius Distribution", 1),
        ("dist_coord_num", np.arange(0, 15, 1), "Coordination number", "D: Coordination Number", 2)
    ]
    
    for key, bins, xlabel, title, ax_idx in configs:
        ax = axes[ax_idx]
        
        # 1. Original Structure (Single Line, no error bars)
        orig_data = original_res[key]
        orig_hist, orig_edges = np.histogram(orig_data, bins=bins)
        orig_freq = (orig_hist / len(orig_data)) * 100
        bin_centers = (orig_edges[:-1] + orig_edges[1:]) / 2
        
        ax.plot(bin_centers, orig_freq, 'ks-', label='Original Structure', linewidth=1.5)
        
        # 2. Reconstructed Structures (Mean Line + Error Bars)
        # Extract the list of raw data arrays from the result dictionaries
        recon_data_list = [r[key] for r in recon_results_list]
        
        recon_mean, recon_ci, _ = get_histogram_stats(recon_data_list, bins)
        
        # Plot with Error Bars
        ax.errorbar(bin_centers, recon_mean, yerr=recon_ci, fmt='ro-', 
                    label='Reconstructed', capsize=4, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Frequency (%)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Figure_3_Comparison.png', dpi=300)
    plt.show()

    generate_figure_3(original_results, recon_results_list)

if __name__ == "__main__":

    # load the full 3D stack, not just a slice
    print("Analyzing Original Structure...")
    original_stack = io.imread('originaldata.tif')
    # convert original stack to binary if it isn't already
    original_binary = (original_stack > filters.threshold_otsu(original_stack)).astype(bool) 
    
    original_results = evaluate_structure(original_binary, voxel_size=3e-9)
    
    recon_results_list = []
    num_reconstructions = 0
    
    for i in range(num_reconstructions):
        filename = f'reconstructed_structure_{i+1}.npy'
        print(f"Analyzing {filename}...")
        
        recon_stack = np.load(filename)
        res = evaluate_structure(recon_stack, voxel_size=3e-9)
        recon_results_list.append(res)

    stats_summary = calculate_statistics(recon_results_list)
    print_table_comparison(original_results, stats_summary)

    with open('finaleval.txt', 'w') as f:
        f.write(print_table_comparison(original_results, stats_summary))
        f.write(generate_figure_3(original_results, recon_results_list))