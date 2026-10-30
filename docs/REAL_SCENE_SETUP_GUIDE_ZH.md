# 真机实验环境搭建指南（one_leg，Stage 3–5）

这份指南告诉你怎样摆好桌子、零件、背景、灯光和两台相机，让真实场景和仿真对得上。只有对得上，下面三步才能顺利进行：

- **Stage 3**：用你的相机参数重新渲染 D_synth。
- **Stage 4**：采集 D_real。
- **Stage 5**：部署并评估 π_real。

机器人控制本身（FR3 + Polymetis、teleop 脚本）见 [FR3_ORIGINAL_CODE_TELEOP_TEST.md](FR3_ORIGINAL_CODE_TELEOP_TEST.md) 和 [REAL_ROBOT_FR3_GUIDE.md](REAL_ROBOT_FR3_GUIDE.md)。各 stage 的命令见 [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)。

文中所有数值都来自代码：仿真场景在 [furniture_sim_env.py](../furniture-bench/furniture_bench/envs/furniture_sim_env.py)，渲染场景在 [isaac_lab_rerender.py](../src/sim2real/isaac_lab_rerender.py)，零件初始位姿在 [config.py](../furniture-bench/furniture_bench/config.py)。论文本身没有给出场景尺寸。

---

## 0. 核心原则

1. **一切以机器人基座坐标系为准。** 策略的状态和动作都是指尖在基座系下的位姿，仿真里的零件、固定条和相机也都定义在这个坐标系里。真实场景只要在基座系下和仿真一致，桌子在房间里放在哪里都无所谓。
2. **相机要对齐到毫米级，而且装好以后不能再动。** 渲染时对相机位姿做的随机化很小：前视相机的位置标准差 2.5 mm、角度 0.5°，腕部相机 0.75 mm、0.25°（`random_camera_pose`，[isaac_lab_rerender.py:554](../src/sim2real/isaac_lab_rerender.py#L554)）。所以渲染相机必须和真实相机**仔细对齐**（§7），不能"量个大概"。
3. **Stage 4 和 Stage 5 用同一个场景。** 桌子、背景、灯光、相机、固定条的标称位置在采集和部署之间都不能变。

| Stage | 对场景的要求 |
|---|---|
| 3 渲染 D_synth | 渲染器里的相机内参、外参与真实相机一致；真实零件和背景的外观与渲染近似（渲染时会随机化颜色和光照） |
| 4 采集 D_real | 零件和固定条的初始位置落在仿真 "low" 随机范围内（§5.3）；相机与 Stage 3 用的完全相同 |
| 5 部署 | 和 Stage 4 完全相同的场景；评估点取自同一个随机范围 |

---

## 1. 坐标系和总体布局

**基座坐标系**：原点在 FR3 基座（link0）安装面的中心；x 轴指向机器人正前方，y 轴指向机器人左侧，z 轴向上。仿真里 `ROBOT_HEIGHT = 0`，也就是**桌面和机器人安装面在同一高度**。

俯视图如下，单位 cm，x 向右，y 向上。前视相机在右侧，面朝机器人。

```
 y
 +121 ─ 绿色背景侧墙 (y=+121, x 从 -50 到 +90) ─────────────────────────────────────────
  +40 ┌───────────────────────────────────── 桌子 130 × 80 ───────────────────────────────┐
      │                                                                                   │
  +20 │                    ══leg4══▶ (装配这条)                                            │
  +17 │                                         ┃━━━━━━━━━━━━━━━━━━┃ ← 侧挡条 (y=+16.5~+18.5)
  +12 │                    ══leg3══▶            ┃   ┌──────────┐   ┃                         │
 背景  │   ╔═══╗                                 ┃   │ tabletop │   ┃ ← 前挡条 x=67~69         │   📷 前视相机
 x=-50│   ║FR3║ (0,0)                           ┃   │ 底面朝上 │   ┃                          │   (112, -6.5, 38.5)
  -12 │   ╚═══╝            ══leg2══▶            ┃   └──────────┘   ┃                         │   朝 -x，下俯 21.5°
  -17 │                                         ┃━━━━━━━━━━━━━━━━━━┃ ← 侧挡条 (y=-16.5~-18.5)
  -20 │                    ══leg1══▶                                                       │
  -40 └───────────────────────────────────────────────────────────────────────────────────┘
     x=-35     0           31.6  40.3   46   52      62.3     69                           95
                                 （桌面 46~62.3，侧挡条 52~69）
             桌子后缘（机器人身后 35 cm）                                          桌子前缘
 ══legN══▶ : 桌腿平躺，箭头 ▶ 指向平端（朝前视相机），螺纹端朝机器人
```

---

## 2. 采购和打印清单

| 项 | 规格 / 说明 |
|---|---|
| 桌子 | 桌面平整，**黑色哑光**。仿真桌子是 130 × 80 cm，和 FurnitureBench 用的 IKEA TOMMARYD 一样大 |
| 机器人 | FR3 + **Franka Hand**（one_leg 需要抓取和拧转） |
| 3D 打印件 | FurnitureBench 的 `square_table`（1 个桌面 + 4 条腿）、`obstacle`（1 根前挡条 + 2 根侧挡条）、腕部相机支架。白色 PLA。下载地址见 [STL Google Drive](https://drive.google.com/drive/folders/1Boj7pyNWklOUVA0ByO0d-J7DM7xfFfRg) |
| 相机 | 2 × RealSense D435；USB 3 线缆，腕部那根留余量或用有源延长线；前视相机用三脚架或万向夹臂 |
| 背景 | 绿色背景布，加框架或支架，要能挡住机器人身后和左右两侧 |
| 灯光 | LED 面板，4600–6000 K，500–1000 lm（FurnitureBench 推荐）；房间避开阳光直射 |
| 小工具 | 双面橡胶胶带或可移除胶（固定挡条）、美纹纸胶带、记号笔、钢尺和卷尺、细线 |

---

## 3. 桌子和机器人

1. **高度**：桌面要和 FR3 安装面平齐，误差控制在 5 mm 以内。如果 FR3 装在独立底座上，就调桌子高度（加垫或用升降桌），不要改仿真。桌面高度不对，策略学到的 z 方向动作在真机上就会抓空或撞桌。
2. **位置**：机器人基座在桌子 y 方向的中线上，离桌子后缘约 35 cm，x 轴朝着桌子长边方向。基座系下的桌面范围大约是 x ∈ [−35, 95] cm，y ∈ [−40, 40] cm。
3. **工作区清空**：x ∈ [25, 80] cm、|y| ≤ 30 cm 范围内只放 one_leg 的零件和挡条。
4. **在桌上画出基座坐标系**：§4.1 介绍用机器人自己当画线工具，这是最准的办法。先画一条 x 轴线（y = 0），再在 x = 30、40、50、60、70 cm 处画刻度。
5. **检查桌面高度**：用 §4.1 的脚本让指尖在工作区四角和中心各碰一次桌面，读出的 z 都应该在 0 ± 5 mm 以内。

---

## 4. 用机器人做尺子（推荐）

手工测量基座中心很难测准，直接让指尖去点目标位置要准得多。下面的脚本需要先装好 Franka Hand，并且 Polymetis 服务器已经启动（[FR3_ORIGINAL_CODE_TELEOP_TEST.md](FR3_ORIGINAL_CODE_TELEOP_TEST.md)）。

### 4.1 标记点脚本（草稿，未在真机上验证）

```python
# scratch/mark_points.py — 在 polymetis 环境里运行；手里始终握着使能器
import numpy as np, torch
from polymetis import RobotInterface

TIP = 0.1034                                   # 法兰 → 指尖（teleop_sm.py 里的 convert_wrist2tip）
DOWN = torch.tensor([1.0, 0.0, 0.0, 0.0])      # 法兰朝下的四元数 (xyzw)
robot = RobotInterface(ip_address="localhost") # 和 teleop 用的是同一个服务器

# 要标记的点（基座系，单位 m）：x 轴刻度 + 零件标称中心（§5.1）
points = [(0.30, 0.0), (0.40, 0.0), (0.50, 0.0), (0.60, 0.0), (0.70, 0.0),
          (0.5415, 0.0),                       # 桌面中心
          (0.3715, -0.20), (0.3715, -0.12), (0.3715, 0.12), (0.3715, 0.20),   # 桌腿原点
          (0.6815, 0.0), (0.6065, -0.175), (0.6065, 0.175)]                  # 挡条中心

for x, y in points:
    for z_tip in (0.05, 0.005):                # 先到 5 cm 高，再降到 5 mm 高
        robot.move_to_ee_pose(torch.tensor([x, y, z_tip + TIP]), DOWN, time_to_go=4.0)
    input(f"在指尖正下方标记 ({x*100:.1f}, {y*100:.1f}) cm，按回车继续")
    robot.move_to_ee_pose(torch.tensor([x, y, 0.05 + TIP]), DOWN, time_to_go=2.0)
```

`move_to_ee_pose` 是 Polymetis 自带的接口，内部先算 IK，再做关节插值。第一次运行时把 `time_to_go` 调大，人站在急停旁边。

### 4.2 读取当前指尖位置

用 SpaceMouse 把指尖移到某个点上（比如相机支架的某个角），再读它的坐标。有两个办法：
- 在你自己的 `teleop_sm_fr3.py` 里取消 `print(f"EE xyz: ...")` 那一行的注释。原始的 `teleop_sm.py` 里有这一行，被注释掉了。
- 或者单独运行下面的脚本：

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from polymetis import RobotInterface
r = RobotInterface(ip_address="localhost")
p, q = r.get_ee_pose()
T = np.eye(4); T[:3, :3] = R.from_quat(q.numpy()).as_matrix(); T[:3, 3] = p.numpy()
off = np.eye(4); off[2, 3] = 0.1034; off[:3, :3] = R.from_euler("z", -45, degrees=True).as_matrix()
print("tip (cm):", ((T @ off)[:3, 3] * 100).round(1))
```

---

## 5. 零件和固定条

### 5.1 标称布局（基座系）

| 物体 | 中心位置 (x, y) cm | 范围 / 朝向 | 高度 |
|---|---|---|---|
| **前挡条** `obstacle_front` | (68.15, 0) | x 67.15–69.15，y −17.5–17.5（35 × 2 cm，沿 y 方向） | 3 cm |
| **侧挡条** `obstacle_side` × 2 | (60.65, ±17.5) | x 52.15–69.15，|y| 16.5–18.5（17 × 2 cm，沿 x 方向） | 3 cm |
| **桌面** `square_table_top` | (54.15, 0) | 16.25 cm 见方，x 46.0–62.3，y −8.1–8.1，边和坐标轴平行；**底面（有四个桌腿孔）朝上** | 3.1 cm |
| **桌腿** leg1 / leg2 / leg3 / leg4 | x 原点 37.15，y = −20 / −12 / +12 / **+20** | 平躺，沿 x 方向：螺纹端在 x ≈ 31.6（朝机器人），平端在 x ≈ 40.3（朝前视相机） | 截面 3 × 3 cm |

几点说明：
- **要装配的是 leg4**，即 y = +20 cm、机器人左侧那条（`OneLeg.should_be_assembled = [(0, 4)]`）。另外三条腿也要摆上，因为渲染场景里五个零件都在（`--num-parts 5`）。
- 三根挡条组成一个**开口朝机器人的 U 形**。内腔范围是 x 52–67 cm、|y| < 16.5 cm，桌面放在里面。
- 仿真里的挡条是简单的长方体。FurnitureBench 打印件可能有拼接口或倒角，按**内侧面**对齐就行：前挡条内侧面在 x = 67.15，侧挡条内侧面在 |y| = 16.5。打印完先量一下尺寸，如果和表里差很多，记录下来。
- 摆好以后可以和 [one_leg.png](../src/assets/calibration/one_leg.png) 对照。那张是 FurnitureBench 的前视参考图，零件上贴了 AprilTag；布局和这里一致。

### 5.2 外观

- **零件上不要贴 AprilTag。** 渲染用的是去掉 tag 的模型（`*_no_tag.usda`），论文附录 III 也专门去掉了 tag。
- **零件用白色 PLA。** 渲染默认是白色（`PART_COLOR_BASE = "white"`），然后在此基础上随机化颜色。论文 Fig. 23–24 显示，零件颜色换成黑色时，需要用颜色随机化过的合成数据才能恢复性能。
- 桌面黑色，背景绿色（§6），和仿真纹理（`table_top_black2.jpeg`、`silk9.jpg`）大致一致即可，渲染时桌面颜色和光照也会随机化。

### 5.3 初始位置的随机范围（low）

数据采集和评估都按论文的 low 设置（论文 Evaluation protocol 一节），偏移都是在水平面内、相对 §5.1 的标称位置：

| 物体 | 偏移范围 |
|---|---|
| 零件（桌面、leg4，以及其余三条腿） | Δx、Δy ∈ [−1.5, +1.5] cm |
| U 形固定条（三根一起移动） | Δx、Δy ∈ [−2, +2] cm |

建议在桌上画网格，方便每次复位：
- **零件网格**：Δ ∈ {−1.5, 0, +1.5} cm 的 3 × 3 = 9 个点。重点是桌面和 leg4，用美纹纸胶带贴出它们的轮廓角标。
- **固定条网格**：Δ ∈ {−2, 0, +2} cm 的 3 × 3 = 9 个点。贴出前挡条两端的位置。
- 固定条要**在一个 episode 里保持不动**（被撞会移位），用双面橡胶胶带或可移除胶粘住，换网格点时再揭下来。

---

## 6. 背景和灯光

- **背景**：仿真里是一个三面的绿色背景：后墙在机器人身后 x = −50 cm 处，两侧墙在 |y| ≈ 121 cm 处，从 x = −50 延伸到 x = +90，高度从地面起约 1.5 m。实际上只要**前视相机画面里桌子以外的地方全是平整的绿色背景**就行，不要有褶皱，也不要露出人、显示器或杂物。
- **灯光**：LED 面板放在桌子一侧（FurnitureBench 放在左侧），色温和亮度见 §2。**Stage 4 和 Stage 5 用同一套灯光**，关掉会变化的光源，比如拉上窗帘。
- **曝光**：原代码对 RealSense 没有做任何设置，用的是默认的自动曝光和自动白平衡。灯光稳定时这样就够了。如果画面明暗时有跳变，可以锁定曝光和白平衡，但 D_real 和部署时必须用同一套设置。

---

## 7. 相机安装和对齐

### 7.1 前视相机

**标称位姿**取自渲染器（[isaac_lab_rerender.py:735-741](../src/sim2real/isaac_lab_rerender.py#L735-L741)），换算到基座系：

| 量 | 值 |
|---|---|
| 位置 | (112.0, −6.5, 38.5) cm，在桌子前缘外约 17 cm，需要三脚架或夹臂 |
| 朝向 | 面朝机器人（−x 方向），下俯 21.5°，无侧倾 |
| 光轴落点 | 桌面上 (14.3, −6.5) cm，靠近机器人基座 |

装好以后的画面应该和 [setup_front.png](../src/assets/calibration/setup_front.png) 差不多：机器人基座在画面上部中间，桌面占画面下方大约 2/3。

**构图检查（重要）**：相机按 640 × 480 采集。处理时整幅缩放到 320 × 240，策略推理时再中心裁剪 224 × 224，训练时在中间 280 像素宽的范围内随机裁剪 224 × 224（[vision.py](../src/common/vision.py)）。换算回原图：

> 策略能看到的区域 ≈ 640 × 480 原图里 **x ∈ [96, 544]、y ∈ [16, 464]** 的范围。

所以 U 形固定条、全部零件、以及指尖在整个工作区内的运动范围，都必须落在这个框里，**而且最好离框边留出余量**。

要求：
- 用夹臂或三脚架**牢牢固定**，线缆用扎带固定住，以免被拉动。
- 尽量接近标称位姿。如果做不到，§7.4 会把渲染器改成你的实际位姿，但离标称越远，和作者的设置差别就越大。

### 7.2 腕部相机

**标称位姿**（[isaac_lab_rerender.py:744-751](../src/sim2real/isaac_lab_rerender.py#L744-L751)），在 `panda_hand` 坐标系下：

| 量 | 值 |
|---|---|
| 位置 | (−5, 0, +4) cm：手爪 −x 侧的平面上，沿手指方向偏 4 cm |
| 朝向 | 视线基本沿手指方向（+z），向手指中心内倾约 20° |

`panda_hand` 的原点在法兰中心，坐标轴相对法兰绕 z 轴转了 −45°；z 轴沿手指方向，手指沿 y 方向开合。

安装要点：
- 用 FurnitureBench 的打印支架，装在手爪**背面**（即 −x 侧），镜头朝向指尖。
- 线缆用三根扎带固定在机械臂上，并留出足够余量。关节转到极限时线不能被拉紧，否则相机会被拽偏。
- 在 home 位姿下，画面下方应该能看到两根指尖，前方能看到桌面和零件。

### 7.3 序列号

把序列号写进 [serials.py](../src/real/serials.py)：`WRIST_CAMERA_SERIAL` 对应 `color_image1`，`FRONT_CAMERA_SERIAL` 对应 `color_image2`。写完以后用手遮住其中一个镜头，确认对应关系。两台相机装反了不会报错，策略会直接变差。

### 7.4 把真实相机写进渲染器（Stage 3 的前提）

**(1) 内参**：读出出厂内参，填进 `front_435_camera_matrix` 和 `wrist_435_camera_matrix`（[isaac_lab_rerender.py:136-149](../src/sim2real/isaac_lab_rerender.py#L136-L149)）：

```python
import pyrealsense2 as rs
for serial in ["<front_serial>", "<wrist_serial>"]:
    cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipe = rs.pipeline(); prof = pipe.start(cfg)
    i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(serial, "fx fy cx cy =", i.fx, i.fy, i.ppx, i.ppy); pipe.stop()
```

注意：`build_camera_params` 只用到 fx、fy 的平均值，cx 和 cy 没有用上。

**(2) 外参初值**，二选一：
- **测量法**：用 §4.2 读出前视相机镜头附近一点的基座系坐标，用卷尺量高度，用量角器或手机水平仪量俯角，得到一个粗略位姿。
- **手眼标定法（更准）**：在手爪上固定一块 ArUco 或棋盘格标定板，让机器人走 15–20 个姿态，用 `cv2.calibrateRobotWorldHandEye` 求出前视相机在基座系下的位姿（eye-to-hand）。腕部相机用 `cv2.calibrateHandEye` 求出它相对法兰的位姿（eye-in-hand），这时标定板固定在桌上。

**(3) 换算成渲染器的格式。** 渲染器用 Isaac/USD 的相机约定：视线沿 −Z，上方向是 +Y，欧拉角是内旋 "XYZ"。世界坐标 = 基座坐标 + (−0.30, 0, 0.415)。OpenCV 的相机约定是 x 右、y 下、z 前。

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
FLIP = np.diag([1, -1, -1])                       # OpenCV 相机系 → USD 相机系

# 前视：T_base_cam 是 OpenCV 约定下相机在基座系的 4×4 位姿
def front_to_rerender(T_base_cam):
    pos = T_base_cam[:3, 3] + np.array([-0.30, 0.0, 0.415])
    eul = R.from_matrix(T_base_cam[:3, :3] @ FLIP).as_euler("XYZ", degrees=True)
    return pos, eul                               # → 填进 cam_pos 和 R.from_euler("XYZ", ...) 的角度

# 腕部：T_flange_cam 是 OpenCV 约定下相机相对法兰(link8)的位姿；panda_hand = 法兰绕 z 转 -45°
def wrist_to_rerender(T_flange_cam):
    T_hand_flange = np.eye(4); T_hand_flange[:3, :3] = R.from_euler("z", 45, degrees=True).as_matrix()
    T = T_hand_flange @ T_flange_cam
    return T[:3, 3], R.from_matrix(T[:3, :3] @ FLIP).as_euler("XYZ", degrees=True)
```

可以先用标称值验证一下：把渲染器里的前视位姿 `(0.82, -0.065, 0.8)`、`[0, 68.5, 90]°` 反算回来，应该得到视线 ≈ (−0.93, 0, −0.37)，也就是朝 −x、下俯 21.5°。

**(4) 叠图对齐（必做）。** 让真实画面和渲染画面在**完全相同的状态**下对比，反复微调外参，直到二者重合：

1. **选一个参考状态**：取一条仿真 rollout 的 pickle，比如 `$DATA_DIR_RAW/raw/diffik/sim/one_leg/rollout/low/success/` 下的任意一个，把它截短到前 3 步，单独放进一个文件夹：
   ```python
   # 在 resip 环境、仓库根目录下运行：python scratch/make_align_pkl.py <src.pkl> scratch/align/align.pkl
   import sys, pickle, numpy as np
   from scipy.spatial.transform import Rotation as R
   from src.visualization.render_mp4 import unpickle_data
   d = unpickle_data(sys.argv[1])
   d["observations"] = d["observations"][:3]; d["actions"] = d["actions"][:2]
   for k in ("rewards", "skills"):
       if k in d: d[k] = d[k][:2]
   pickle.dump(d, open(sys.argv[2], "wb"))
   # 打印需要在真机上复现的状态
   A = np.array([[0,1,0,0.0015],[1,0,0,0],[0,0,-1,0.415],[0,0,0,1.]]); base = np.array([-0.3,0,0.415])
   o = d["observations"][0]
   print("joints:", np.round(o["robot_state"]["joint_positions"], 4).tolist())
   pp = np.asarray(o["parts_poses"]).reshape(-1, 7)
   for name, p in zip(["tabletop","leg1","leg2","leg3","leg4","obstacle_front"], pp):
       T = np.eye(4); T[:3, 3] = p[:3]; T[:3, :3] = R.from_quat(p[3:]).as_matrix()
       print(f"{name:15s} (x, y) = ({(A@T)[0,3]*100-base[0]*100:6.1f}, {(A@T)[1,3]*100:6.1f}) cm")
   ```
   这段脚本假设 parts_poses 的格式是每个零件 7 维（位置 + xyzw 四元数，AprilTag 坐标系），最后 7 维是前挡条。这和渲染器读取它的方式一致。侧挡条跟前挡条一起移动：x 比前挡条小 7.5 cm，y = ±17.5 cm。
2. **真机复现这个状态**：
   - 在 Polymetis 环境里执行 `robot.move_to_joint_positions(torch.tensor(joints))`，`joints` 用上一步打印出来的值。
   - 按打印出的坐标摆好零件和挡条（用 §4.1 的标记点）。
   - 采集一帧前视图和一帧腕部图（640 × 480，rgb8），保存成 `real_front.png` 和 `real_wrist.png`。
3. **渲染这个状态**（不加 `-dr`）：
   ```bash
   python -m src.sim2real.isaac_lab_rerender --headless -i 0 --sub-steps 3 \
     --load-dir scratch/align --furniture one_leg --num-parts 5 --save --save-dir scratch/align_out
   ```
   从输出的 pickle 里取第 0 帧的 `color_image2`（前视）和 `color_image1`（腕部）。
4. **叠图**：
   ```python
   import cv2
   real, sim = cv2.imread("real_front.png"), cv2.imread("sim_front.png")
   edges = cv2.Canny(cv2.cvtColor(sim, cv2.COLOR_BGR2GRAY), 50, 150)
   out = cv2.addWeighted(real, 0.6, sim, 0.4, 0); out[edges > 0] = (0, 0, 255)   # 仿真边缘画成红线
   cv2.rectangle(out, (96, 16), (544, 464), (255, 255, 0), 1)                     # 策略能看到的区域
   cv2.imwrite("overlay_front.png", out)
   ```
5. **调整外参**：前视相机改 `cam_pos` 和欧拉角，腕部相机改 `local_wrist_pos` 和 `local_wrist_rot`。改完重新渲染、重新叠图，直到达标。
   - **达标标准**：在青色框内，桌面边缘、挡条、零件、手指的红线和真实边缘的偏差都在约 5 像素以内。在约 0.7 m 的距离上，1° 的角度误差大约对应 10 像素，所以 5 像素大致相当于渲染随机化的范围。
   - **调整顺序**：先调俯仰和偏航，让画面里的整体位置对上；再调平移，让大小和透视对上；最后调滚转。
6. **保存基准图**：把对齐用的 `real_front.png`、`real_wrist.png` 和关节角存好。§9 每天检查相机有没有移动时会用到。

对齐完成后，就可以用你的相机参数跑 Stage 3 的批量渲染了（[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) 3b–3c）。

---

## 8. Stage 4：采集 D_real

每个 episode 开始前：
1. 机器人回 home 位姿。teleop 脚本会自动执行，并给每个关节加 ±5° 的随机噪声。
2. **复位零件**：从 §5.3 的零件网格里取下一个点，按胶带标记摆好桌面和 leg4，其余三条腿放在标称位置附近。
3. **复位固定条**：按计划取一个固定条网格点（换点时揭开胶带再重新粘好）。
4. 确认灯光一致，画面里没有手、工具或其它杂物，相机没有被碰过。
5. 运行 teleop，完成后按 `t` 保存（`n` 放弃）。

采集计划：
- 论文用了 10–40 条示教。
- 让零件网格和固定条网格的点都被覆盖到，比如零件 9 个点 × 固定条在中心，再加上固定条 9 个点 × 零件在中心。
- 每条示教都要从网格点出发，不要每次都从同一个位置开始。

采完之后：
- 用 [DATA_GUIDE.md](DATA_GUIDE.md) §9 的工具查看视频，确认相机顺序和颜色正常。
- 用 notebook `07_real_vs_sim_state_action_space` 检查**真实动作是否落在仿真动作的分布里**（论文 Fig. 19）。如果没有，最常见的原因是零件或挡条的位置和 §5.1 差得太多，或者桌面高度不对。

---

## 9. Stage 5：部署和评估

- 场景和 Stage 4 完全相同。部署前先做 §9.1 的相机检查。
- **评估**：按论文的做法做两组，每组 10 次。
  - 第一组只改变零件位置，从零件网格里取点，固定条在中心。
  - 第二组只改变固定条位置，从固定条网格里取点，零件在中心。
  - 每次按阶段记录成功与否：对准角落、抓取、插入、拧紧。
  - 比较不同策略时，使用同一组网格点。

### 9.1 每次开工前检查（约 2 分钟）

1. 机器人移到 §7.4 保存的那组关节角，按对齐时的布局摆好零件。
2. 拍一张前视图，和 `real_front.png` 做差，或者直接叠图看。
3. 如果桌面、挡条这些静止物体的位移超过约 3 像素，说明**相机被动过了**。这时有两个选择：
   - 把相机调回原位，直到叠图重合。
   - 或者重新做 §7.4 并重新渲染。但 Stage 4 已经采集的数据是旧视角下的，混用会降低效果。
4. 腕部相机同样用 `real_wrist.png` 检查。重点看指尖在画面里的位置有没有变。

---

## 10. 常见问题

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 叠图怎么调都对不上，越调越偏 | 内参没换，或者把 OpenCV 和 USD 的相机约定搞混了 | 检查 §7.4 (1)，以及 (3) 里的 `FLIP` |
| 远处对上了，近处没对上 | 平移或焦距不对 | 先确认内参，再调相机的前后距离 |
| 真实动作的分布比仿真偏高或偏低 | 桌面不在安装面高度 | §3 第 1、5 步 |
| 真实动作在 x 或 y 方向整体偏移 | 零件或挡条整体偏离标称位置，或者基座系标记画错了 | 用 §4.1 重新标记 |
| 策略在真机上总往同一个方向偏 | 相机在采集后被移动过 | §9.1 |
| 腕部画面时好时坏 | 线缆拉扯导致支架松动 | 加固支架，给线缆留出余量 |

---

## 11. 还没有核实的地方

- **打印件的实际尺寸**：§5.1 用的是仿真网格的尺寸。打印出来以后实测一下，有差别就按内侧面对齐。
- **腕部相机支架的安装方向**：FurnitureBench 文档说装在"背面、镜头朝向指尖"。我是根据渲染参数推断出这就是 `panda_hand` 的 −x 侧的，装好以后请用 §7.4 的叠图确认。
- **§4 和 §7.4 的脚本**：都是草稿，没有在真机上运行过。特别是 `move_to_ee_pose` 的参数，请以你装的那个 Polymetis 版本为准。
