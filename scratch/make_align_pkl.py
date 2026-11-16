
import sys, pickle, numpy as np
from scipy.spatial.transform import Rotation as R
from src.visualization.render_mp4 import unpickle_data
d = unpickle_data(sys.argv[1])
d["observations"] = d["observations"][:3]; d["actions"] = d["actions"][:2]
for k in ("rewards", "skills"):
    if k in d: d[k] = d[k][:2]
pickle.dump(d, open(sys.argv[2], "wb"))
A = np.array([[0,1,0,0.0015],[1,0,0,0],[0,0,-1,0.415],[0,0,0,1.]]); base = np.array([-0.3,0,0.415])
o = d["observations"][0]
print("joints:", np.round(o["robot_state"]["joint_positions"], 4).tolist())
pp = np.asarray(o["parts_poses"]).reshape(-1, 7)
for name, p in zip(["tabletop","leg1","leg2","leg3","leg4","obstacle_front"], pp):
    T = np.eye(4); T[:3, 3] = p[:3]; T[:3, :3] = R.from_quat(p[3:]).as_matrix()
    print(f"{name:15s} (x, y) = ({(A@T)[0,3]*100-base[0]*100:6.1f}, {(A@T)[1,3]*100:6.1f}) cm")
