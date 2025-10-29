import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.point_clouds import PointClouds
from plyfile import PlyData

def add_keypoints(path, viewer, thisname, color=(1.0, 0.0, 0.0, 1)):
    # Load keypoints
    keypoints = np.load(path)

    keypoints = keypoints[..., :3]

    keypoints_pc = PointClouds(
        keypoints,
        name=thisname,
        point_size=3.0,
        color=color
    )
    viewer.scene.add(keypoints_pc)
    return
    

if __name__ == "__main__":
    # Set the path to your SMPL models
    C.smplx_models = "smpl_models/"
    take="_c2_a5_2"
    keypoints_path = "observations/vitpose"+take+""+"/vitpose/keypoints_3d/full-keypoints-3d.npy"
    smplseq_path="observations/vitpose"+take+"/vitpose/fit-smplx/smplx-params.npz"
    smplkeypoints_path = "observations/vitpose"+take+""+"/vitpose/keypoints_3d/smpl-keypoints-3d.npy"
    #smplhkeypoints_path = "observations/vitpose"+take+""+"/vitpose/keypoints_3d/smplh-keypoints-3d.npy"
    #smplxkeypoints_path = "observations/vitpose"+take+""+"/vitpose/keypoints_3d/smplx-keypoints-3d.npy"
    


    data = np.load(smplseq_path)
    
    # smplx parameters
    body_pose = data['body_pose']           # (419, 63)
    print(body_pose.shape)
    global_orient = data['global_orient']   # (419, 3)
    betas = data['betas']                   # (419, 11)
    transl = data['transl']                 # (419, 3)
    left_hand_pose = data['left_hand_pose'] # (419, 12) - PCA components
    right_hand_pose = data['right_hand_pose'] # (419, 12) - PCA components
    
    print(f"Loaded sequence with {body_pose.shape[0]} frames")
    
    # Create SMPL-X layer with PCA hand pose support
    smpl_layer = SMPLLayer(
        model_type="smplx", 
        gender="neutral", 
        device=C.device,
        age="kid",
        kid_template_path=r"C:\Users\Rui\Vorlesungskript\Master\Thesis\test\smpl_models\smplx\smplx_kid_template.npy",
        use_pca=True, 
        num_pca_comps=12,  
        flat_hand_mean=False
    )
    smpl_layer.num_betas += 1
    
    # Create SMPL-X sequence
    # When use_pca=True, the layer expects 12-dimensional hand poses
    smpl_sequence = SMPLSequence(
        poses_body=body_pose,
        poses_root=global_orient,
        betas=betas,
        trans=transl,
        poses_left_hand=left_hand_pose,
        poses_right_hand=right_hand_pose,
        smpl_layer=smpl_layer,
        name="SMPL-X Sequence",
    )
    
    # Load floor point cloud from PLY file
    ply_data = PlyData.read("floor/floor_c1_a3.ply")
    vertices = ply_data['vertex']
    floor_points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Create point cloud (add a frame dimension if needed)
    floor_pc = PointClouds(floor_points[np.newaxis, :, :], name="MyFloor", point_size=2.0)
    
    # Load keypoints
    keypoints = np.load(keypoints_path)

    keypoints = keypoints[..., :3]

    keypoints_pc = PointClouds(
        keypoints,
        name="FULL Keypoints",
        point_size=3.0,
        color=(1.0, 1.0, 0.0, 1)
    )
    print(keypoints.shape)

    
    # Display in viewer
    v = Viewer()
    #v.scene.add(keypoints_pc)
    add_keypoints(smplkeypoints_path, v, "SMPL Keypoints")
    #add_keypoints(smplxkeypoints_path, v, "SMPLX Keypoints")
    v.scene.add(smpl_sequence)
    v.scene.add(floor_pc)

    v.run()