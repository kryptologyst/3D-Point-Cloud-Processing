"""Streamlit demo for 3D point cloud processing."""

import os
import tempfile
from typing import Optional

import numpy as np
import streamlit as st
import torch
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.pointnet import PointNet
from src.utils.device import get_device
from src.utils.metrics import chamfer_distance, earth_mover_distance


# Page config
st.set_page_config(
    page_title="3D Point Cloud Processing",
    page_icon="🔺",
    layout="wide"
)

# Title
st.title("3D Point Cloud Processing Demo")
st.markdown("Upload a point cloud file or generate synthetic shapes for classification and analysis.")


@st.cache_resource
def load_model():
    """Load the trained model."""
    device = get_device("auto")
    model = PointNet(num_classes=4, task="classification")
    
    # Try to load checkpoint
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device, True
    else:
        # Return untrained model for demo
        model.eval()
        return model, device, False


def preprocess_point_cloud(points: np.ndarray, num_points: int = 1024) -> torch.Tensor:
    """Preprocess point cloud for model input."""
    # Sample points if needed
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        # Pad with random points
        pad_size = num_points - len(points)
        pad_points = np.random.normal(0, 0.1, (pad_size, 3))
        points = np.vstack([points, pad_points])
    
    # Normalize points
    points = points - np.mean(points, axis=0)
    points = points / np.std(points, axis=0)
    
    return torch.FloatTensor(points).unsqueeze(0)


def visualize_point_cloud(points: np.ndarray, title: str = "Point Cloud") -> go.Figure:
    """Create 3D visualization of point cloud."""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=600,
        height=500
    )
    
    return fig


def generate_synthetic_shape(shape_type: str, num_points: int = 1024) -> np.ndarray:
    """Generate synthetic 3D shapes."""
    if shape_type == "Cube":
        # Generate points on cube faces
        points = []
        for _ in range(num_points // 6):
            # Random face
            face = np.random.randint(0, 6)
            if face == 0:  # z = 0
                x, y, z = np.random.uniform(-0.5, 0.5, 3)
                z = 0
            elif face == 1:  # z = 1
                x, y, z = np.random.uniform(-0.5, 0.5, 3)
                z = 1
            elif face == 2:  # x = 0
                x, y, z = np.random.uniform(-0.5, 0.5, 3)
                x = 0
            elif face == 3:  # x = 1
                x, y, z = np.random.uniform(-0.5, 0.5, 3)
                x = 1
            elif face == 4:  # y = 0
                x, y, z = np.random.uniform(-0.5, 0.5, 3)
                y = 0
            else:  # y = 1
                x, y, z = np.random.uniform(-0.5, 0.5, 3)
                y = 1
            points.append([x, y, z])
        
    elif shape_type == "Sphere":
        points = []
        for _ in range(num_points):
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.random.uniform(0, np.pi)
            x = 0.5 * np.sin(theta) * np.cos(phi)
            y = 0.5 * np.sin(theta) * np.sin(phi)
            z = 0.5 * np.cos(theta)
            points.append([x, y, z])
    
    elif shape_type == "Cylinder":
        points = []
        for _ in range(num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0, 1)
            x = 0.5 * np.cos(theta)
            y = 0.5 * np.sin(theta)
            points.append([x, y, z])
    
    elif shape_type == "Cone":
        points = []
        for _ in range(num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0, 1)
            r = 0.5 * (1 - z)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
    
    return np.array(points)


def main():
    """Main demo function."""
    # Load model
    model, device, is_trained = load_model()
    
    if not is_trained:
        st.warning("⚠️ No trained model found. Using untrained model for demonstration.")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Input method
    input_method = st.sidebar.radio(
        "Input Method",
        ["Upload File", "Generate Synthetic"]
    )
    
    if input_method == "Upload File":
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Point Cloud",
            type=['ply', 'pcd', 'obj'],
            help="Upload a .ply, .pcd, or .obj point cloud file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Load point cloud
                pcd = o3d.io.read_point_cloud(tmp_path)
                points = np.asarray(pcd.points)
                
                if len(points) == 0:
                    st.error("Error: Could not load point cloud from file.")
                    return
                
                st.success(f"✅ Loaded point cloud with {len(points)} points")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
            finally:
                os.unlink(tmp_path)
    
    else:
        # Generate synthetic shape
        shape_type = st.sidebar.selectbox(
            "Shape Type",
            ["Cube", "Sphere", "Cylinder", "Cone"]
        )
        
        num_points = st.sidebar.slider(
            "Number of Points",
            min_value=100,
            max_value=2048,
            value=1024,
            step=100
        )
        
        if st.sidebar.button("Generate Shape"):
            points = generate_synthetic_shape(shape_type, num_points)
            st.success(f"✅ Generated {shape_type} with {len(points)} points")
    
    # Process point cloud if available
    if 'points' in locals():
        # Display point cloud
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Point Cloud Visualization")
            fig = visualize_point_cloud(points)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Analysis")
            
            # Basic statistics
            st.write("**Basic Statistics:**")
            st.write(f"- Number of points: {len(points)}")
            st.write(f"- Bounding box: {np.min(points, axis=0)} to {np.max(points, axis=0)}")
            st.write(f"- Mean: {np.mean(points, axis=0)}")
            st.write(f"- Std: {np.std(points, axis=0)}")
            
            # Classification
            if st.button("Classify Point Cloud"):
                with st.spinner("Classifying..."):
                    # Preprocess points
                    input_tensor = preprocess_point_cloud(points)
                    input_tensor = input_tensor.to(device)
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(outputs, dim=1).item()
                    
                    # Class names
                    class_names = ["Cube", "Sphere", "Cylinder", "Cone"]
                    
                    st.write("**Classification Results:**")
                    st.write(f"Predicted class: **{class_names[predicted_class]}**")
                    
                    # Show probabilities
                    st.write("**Class Probabilities:**")
                    for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
                        st.write(f"- {class_name}: {prob:.3f}")
            
            # Point cloud processing options
            st.subheader("Point Cloud Processing")
            
            # Downsampling
            if st.button("Apply Voxel Downsampling"):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                voxel_size = st.slider("Voxel Size", 0.01, 0.1, 0.05, 0.01)
                downsampled_pcd = pcd.voxel_down_sample(voxel_size)
                downsampled_points = np.asarray(downsampled_pcd.points)
                
                st.write(f"Downsampled from {len(points)} to {len(downsampled_points)} points")
                
                # Update visualization
                fig = visualize_point_cloud(downsampled_points, "Downsampled Point Cloud")
                st.plotly_chart(fig, use_container_width=True)
            
            # Plane segmentation
            if st.button("Apply Plane Segmentation"):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                distance_threshold = st.slider("Distance Threshold", 0.01, 0.1, 0.02, 0.01)
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=3,
                    num_iterations=1000
                )
                
                inlier_cloud = pcd.select_by_index(inliers)
                outlier_cloud = pcd.select_by_index(inliers, invert=True)
                
                st.write(f"Plane segmentation: {len(inliers)} inliers, {len(outlier_cloud.points)} outliers")
                
                # Visualize segmentation
                inlier_points = np.asarray(inlier_cloud.points)
                outlier_points = np.asarray(outlier_cloud.points)
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Inliers (Plane)", "Outliers"),
                    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]]
                )
                
                fig.add_trace(
                    go.Scatter3d(
                        x=inlier_points[:, 0],
                        y=inlier_points[:, 1],
                        z=inlier_points[:, 2],
                        mode='markers',
                        marker=dict(size=2, color='red'),
                        name="Inliers"
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter3d(
                        x=outlier_points[:, 0],
                        y=outlier_points[:, 1],
                        z=outlier_points[:, 2],
                        mode='markers',
                        marker=dict(size=2, color='blue'),
                        name="Outliers"
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
