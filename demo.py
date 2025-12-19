import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # <--- NEW IMPORT
from scipy.ndimage import distance_transform_edt
from utils import task2_generate_3d 

st.set_page_config(page_title="2D→3D Membrane Reconstruction", layout="wide")

st.title("🔬 Interactive Membrane Reconstruction")
st.markdown("""
**Hackathon Demo:** Adjust the statistical parameters below to generate a new 3D membrane structure in real-time.
This demonstrates the underlying generative model's responsiveness.
""")

# --- Sidebar ---
st.sidebar.header("Model Parameters")

sigma_1 = st.sidebar.slider("σ₁ (Skin Layer Pore Size)", 1.0, 10.0, 3.0, 0.1, help="Controls the size of small pores at the top")
sigma_2 = st.sidebar.slider("σ₂ (Substructure Pore Size)", 10.0, 30.0, 18.0, 0.5, help="Controls the size of large voids at the bottom")
threshold = st.sidebar.slider("Porosity Threshold", 0.3, 0.7, 0.5, 0.01, help="Cutoff value. Lower = More Solid, Higher = More Pores")
r_struct = st.sidebar.slider("Smoothing Radius (Morphology)", 1, 5, 2, 1, help="Radius of spherical closing element")

if st.sidebar.button("🚀 Generate Structure", type="primary"):
    with st.spinner("Simulating Phase Separation..."):
        # Generate a smaller block for instant feedback
        vol = task2_generate_3d(sigma_1, sigma_2, threshold, r_struct, size=(256, 256, 128))
        st.session_state.vol = vol
        st.session_state.generated = True

# --- Main Display ---
if 'generated' in st.session_state:
    vol = st.session_state.vol
    
    # Layout for 2D Stats
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Top View (Skin)")
        fig, ax = plt.subplots()
        ax.imshow(vol[:, :, 10], cmap='gray_r') # Invert so pores are dark
        ax.axis('off')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Side Cross-Section")
        fig, ax = plt.subplots()
        # Transpose to show depth downwards
        ax.imshow(vol[:, 128, :].T, cmap='gray_r', aspect='auto')
        ax.set_ylabel("Depth (z)")
        ax.set_xticks([])
        st.pyplot(fig)

    with col3:
        st.subheader("Real-time Properties")
        # Invert logic: vol==0 usually means pore in these binary masks, 
        # but check your utils. Assuming 0 is Void here:
        porosity = np.sum(vol == 0) / vol.size
        
        # Fast pore size calc
        dt = distance_transform_edt(vol == 0)
        # Avoid mean of empty array warning
        if np.any(dt > 0):
            avg_pore = np.mean(dt[dt > 0]) * 2 * 3 # 3nm resolution
        else:
            avg_pore = 0.0
        
        st.metric("Porosity", f"{porosity:.1%}")
        st.metric("Est. Mean Pore Size", f"{avg_pore:.1f} nm")
        
        st.success("Structure Generated Successfully")

    st.markdown("---")
    
    # --- NEW 3D VISUALIZATION SECTION ---
    st.subheader("🧊 Interactive 3D Iso-Surface")
    st.caption("Rotate, zoom, and pan to inspect the pore connectivity. (Downsampled for browser performance)")

    # 1. Downsample the volume (Essential for performance!)
    # We take every 4th voxel. Reduces data size by factor of 64 (4*4*4).
    # 256/4 = 64 pixels wide.
    STRIDE = 4 
    vol_small = vol[::STRIDE, ::STRIDE, ::STRIDE]

    # 2. Create coordinates grid
    X, Y, Z = np.mgrid[0:vol_small.shape[0], 0:vol_small.shape[1], 0:vol_small.shape[2]]

    # 3. Create Plotly Isosurface
    # Assuming '0' is pore and '1' is solid (or vice-versa). 
    # This draws a 'skin' around the boundary between 0 and 1.
    fig_3d = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol_small.flatten(),
        isomin=0.5, # Threshold between 0 and 1
        isomax=0.5,
        surface_count=1, # Draw exactly one surface layer
        colorscale='Spectral', # Cool looking colors
        caps=dict(x_show=False, y_show=False), # Hide the ends so it looks like a cut-out
        showscale=False
    ))

    # Update camera angle for a cool initial view
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data' # Keep proportions correct
        ),
        margin=dict(l=0, r=0, b=0, t=0), # Remove whitespace
        height=500
    )

    st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.info("👈 Click 'Generate Structure' to start the simulation.")

st.markdown("---")
st.caption("Generated using Gaussian Random Fields with Depth-Dependent Weighting.")