import numpy as np
import porespy as ps
import openpnm as op
import scipy.ndimage as spim
from skimage import io
from skimage import filters

import scipy.sparse.csgraph as csgraph
from scipy import sparse

def evaluate_structure(im, voxel_size=3e-9):
    im = im.astype(bool)
    
    porosity_total = im.sum() / im.size
    labels = spim.label(im)[0]
    largest_cluster = labels == np.argmax(np.bincount(labels[labels > 0]))
    porosity_connected = largest_cluster.sum() / im.size

    print("  > Extracting pore network ")
    net_ext = ps.networks.snow2(im, voxel_size=voxel_size, boundary_width=0)
    
    try:
        net = op.io.network_from_porespy(net_ext.network)
    except AttributeError:
        net = op.network.Network()
        net.update(net_ext.network)

    # --- Geometry Setup ---
    net['pore.diameter'] = net['pore.inscribed_diameter']
    net['throat.diameter'] = net['throat.inscribed_diameter']
    

    # Get pore & throat geometry (for stats)
    pore_radii = net['pore.diameter'] / 2
    throat_radii = net['throat.diameter'] / 2
    conns = net['throat.conns']
    coordination_num = np.bincount(conns.flatten(), minlength=net.Np)

    return {
        "mean_pore_radius_nm": np.mean(pore_radii) * 1e9,
        "mean_throat_radius_nm": np.mean(throat_radii) * 1e9,
        "mean_coord_num": np.mean(coordination_num),
        "porosity_total": porosity_total,
        "porosity_connected": porosity_connected,
        # "tortuosity":,
        # "permeability_LMH_psi":
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



