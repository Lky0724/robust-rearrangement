# Test: Original Teleop Code on a Franka Research 3

A step-by-step test of whether the authors' **original** real-robot stack (Polymetis + `rdt` + [src/real/teleop_sm.py](../src/real/teleop_sm.py)) can run SpaceMouse teleop on a **Franka Research 3 (FR3)**, with as few code changes as possible.

The test runs in two stages:

1. **Teleop without cameras** (Phases 1–10). A small switch in `teleop_sm.py` (`--no_cameras`, Phase 8) skips the RealSense cameras and saves black placeholder images in their place. You test the robot, the SpaceMouse and the data format before you have cameras.
2. **Add the cameras** (Phase 11), once you have two RealSense D435s. Everything from stage 1 stays; you mount the cameras, enter their serials, and drop `--no_cameras`.

Each phase ends in a **✅ go / no-go check**. Don't start a phase until the previous check passes. If a check fails and the troubleshooting table (§13) doesn't fix it, stop and switch to the replacement robot layer in [REAL_ROBOT_FR3_GUIDE.md](REAL_ROBOT_FR3_GUIDE.md). Most of the work here (`rdt`, cameras, SpaceMouse, checks on the collected data) carries over to that plan.

**How to read the commands**

- Lines starting with `sudo` need your password.
- `# terminal N` means "run this in terminal window N". Keep each terminal open once it's started.
- `(polymetis-local)` means the command runs in that conda env (`conda activate polymetis-local` first). `(resip)` means the main ResiP env.
- **Path A** = you have a real-time (PREEMPT_RT) kernel. **Path B** = you don't. Where the two differ, both are given. §0.3 helps you choose.

---

## 0. Overview

### 0.1 What has to work, and where it can break

```
SpaceMouse ─► rdt Spacemouse ─┐
keyboard   ─► rdt Keyboard ───┤
                              ▼
               src/real/teleop_sm.py ──── 2× D435 via rdt.image.factory (rgb8, 640×480, 30 fps)
                              │             (stage 1: --no_cameras → black placeholder images)
                              │  convert_tip2wrist → DiffIKWrapper.update_desired_ee_pose(T)
                              │     (rdt: q* = q + J⁺·e, computed ONCE per 10 Hz command, client side)
                              │  → RobotInterface.update_desired_joint_positions(q*)
                              │ gRPC (localhost:50051 arm, :50052 gripper)
                              ▼
               Polymetis server: launch_robot.py / launch_gripper.py
                 joint impedance @ 1 kHz (Kq, Kqd from teleop_sm.py)
                              │ libfranka  ◄── ⚠ must be a version your FR3 accepts (≥ 0.10)
                              │            ◄── ⚠ refuses to run without an RT kernel unless patched (§0.3)
                              ▼
               FR3 + Franka Hand (FCI)
```

| Risk | Why | Checked in |
|---|---|---|
| **Polymetis vs. FR3** | The conda `polymetis` package bundles libfranka 0.9.x, which an FR3 rejects. Polymetis has to be **built from source** against a newer libfranka, and the libfranka API changed after 0.9. No maintained Polymetis officially supports the FR3 | Phases 4–5 |
| **Real-time kernel** | libfranka throws an error on a non-RT kernel unless told to ignore it. Without RT, the 1 kHz loop also sometimes misses deadlines and the robot stops | Phases 1, 3, 4 |
| `rdt` | It's public, but only the **`lars-dev`** branch has every module `teleop_sm.py` imports. Its `DiffIKWrapper` constructor doesn't match the call in `teleop_sm.py` | Phases 6, 8 |
| FR3 joint limits | Polymetis' safety limits are set for the Panda. Some FR3 limits are narrower (joint 6 especially) | Phase 5 |
| Everything else | SpaceMouse, keyboard and data format are the same as on the Panda | Phases 7, 9, 10 |
| Cameras | Same as on the Panda, but added last, once you have them | Phase 11 |

### 0.2 Machine layout

Use **one Ubuntu PC**, wired directly to the FR3. It runs the Polymetis server *and* `teleop_sm.py`. Teleop needs no GPU. With one machine, one conda env and one Polymetis build, the client and server versions always match.

The steps below assume this folder layout (change the paths if yours differ):

```
~/ResiP          this repo
~/fr3_test/      everything this test downloads or builds
  libfranka/     standalone libfranka, only for the Phase 3 check
  fairo/         Polymetis source and build
  improbable_rdt/  the authors' rdt tools
```

### 0.3 Do I need a real-time kernel?

**No, not for this test, but expect occasional stops.** The FR3 expects a new command every 1 ms. A real-time (PREEMPT_RT) kernel guarantees that. A normal Ubuntu kernel usually manages it, but sometimes it's late, and the robot then stops with a `communication_constraints_violation` error. The stop is safe: the arm halts, you clear the error and restart. It's annoying during demo collection, and acceptable for checking whether the stack works.

Check which kernel you have:
```bash
uname -a
```
- Contains `PREEMPT_RT` → **Path A**.
- Contains `PREEMPT_DYNAMIC` or `PREEMPT` without `_RT` (for example the stock Ubuntu 24.04 `-generic` kernel) → **Path B**.

What changes on Path B:

| Where | Path A (RT kernel) | Path B (no RT kernel) |
|---|---|---|
| §1.3 RT permissions | Do it | Do it too. It still improves timing |
| §1.4 Optional `preempt=full` | Skip | Recommended |
| §3.2 libfranka check | Run the examples as they are | Patch `communication_test` and `echo_robot_state` first (one line each) |
| §4.3 Polymetis | No patch | Patch `franka_panda_client.cpp` (one line) |
| §5.2 Launch | Default (asks for `sudo`) | Add `use_real_time=false` |

The patch tells libfranka `RealtimeConfig::kIgnore`. By default (`kEnforce`), libfranka refuses to start a control loop on a non-RT kernel with:
```
libfranka: Running kernel does not have realtime capabilities.
```
With `kIgnore` it still tries to give its thread the highest priority, but it carries on if the kernel isn't RT.

If you later collect many demos and the stops become a problem, install an RT kernel (§1.2) and switch to Path A. Nothing else needs to be redone, except removing `use_real_time=false` from the launch command.

---

## 1. Prepare the PC

### 1.1 Gather information and hardware

Before starting, have:

- [ ] **FR3 system version.** Open Desk in a browser (§2), go to *Settings → System*, and write down the system version, for example `5.5.0`.
- [ ] **FR3 IP address** for the FCI connection. This is in Desk under *Settings → Network*. Franka's docs use `172.16.0.2`, and that value is used below. Replace it if yours differs.
- [ ] **Desk login** (user and password).
- [ ] **External enabling / activation device**, plugged into the robot.
- [ ] **SpaceMouse** (wired, or wireless with its USB receiver).
- [ ] *(Stage 2 only, not needed until Phase 11)* **Two Intel RealSense D435** cameras with **USB 3** cables, and rigid mounts (wrist and front).
- [ ] **An Ethernet port on the PC** for the robot. Use a built-in port if you have one. USB-Ethernet adapters often add latency and lost packets. Find your ports with `ip -br link`. Built-in ports are usually named `enp…`/`eno…`; USB adapters are usually `enx…`.

**Pick the libfranka version.** Open Franka's compatibility table (search "libfranka compatibility" on the Franka Robotics docs site) and find your FR3 system version. Choose the **lowest** libfranka version that supports it. Polymetis was written for 0.9, so the fewer API changes since then, the more likely it compiles. FR3 needs at least `0.10.0`. Save your choice so later commands can use it:

```bash
echo 'export FR3_IP=172.16.0.2'        >> ~/.bashrc   # your robot's IP
echo 'export LIBFRANKA_VER=0.10.0'     >> ~/.bashrc   # your chosen libfranka version
source ~/.bashrc
```

### 1.2 Kernel (Path A only)

Skip this on Path B.

On Ubuntu 22.04 or 24.04, the easiest RT kernel is Canonical's:
```bash
sudo pro attach            # needs a free Ubuntu Pro token from ubuntu.com/pro
sudo pro enable realtime-kernel
sudo reboot
uname -a                   # must now contain PREEMPT_RT
```
Alternatively, build one as described in Franka's installation docs.

⚠ The NVIDIA driver often doesn't install on an RT kernel. The old kernel stays installed, so you can pick it under *Advanced options for Ubuntu* in the GRUB boot menu when you need the GPU (training, sim), and boot the RT kernel for the robot.

### 1.3 Real-time permissions (both paths)

These let your programs give their control thread real-time priority without `sudo`, so libfranka's 1 kHz thread runs ahead of everything else on the PC. Without them, `ulimit -r` prints `0`, and libfranka can't raise its priority.

**1. Note your current memory-lock limit** (you need it in step 3):
```bash
ulimit -l
```

**2. Create a `realtime` group and add yourself to it:**
```bash
sudo addgroup realtime
sudo usermod -a -G realtime $USER      # -a = append; without it you'd be removed from your other groups
```

**3. Allow that group real-time priority.** This appends two lines to `/etc/security/limits.conf`, which Linux applies at every login:
```bash
sudo tee -a /etc/security/limits.conf <<'EOF'
@realtime soft rtprio 99
@realtime hard rtprio 99
EOF
```
- `@realtime` = every member of the group `realtime`.
- `rtprio 99` = they may give their threads real-time priorities up to 99, the maximum.
- `hard` is the ceiling; `soft` is what you get by default. Setting both to 99 means "available right away".

**Only if step 1 printed a number below `102400`** (for example `8192`), also raise the memory-lock limit to 100 MB. A real-time thread locks its memory into RAM so it never waits for swap:
```bash
sudo tee -a /etc/security/limits.conf <<'EOF'
@realtime soft memlock 102400
@realtime hard memlock 102400
EOF
```
If step 1 printed `unlimited` or a larger number, skip this. Adding the lines would **lower** your limit and could break other programs that lock memory.

> Franka's installation docs also list `@realtime soft/hard priority 99`. Leave those lines out. `priority` sets the normal "nice" value of every process in your login session, whose valid range is -20 to 19. Out of range, it most likely ends up as 19, the lowest normal priority, slowing your whole desktop. libfranka doesn't need it.

**4. Log out completely and log back in** (reboot if unsure). The limits only apply to new logins. Then check:
```bash
groups          # must include: realtime
ulimit -r       # must print: 99
ulimit -l       # must print the same value as in step 1 (or 102400 if you raised it)
nice            # must print: 0
```

**Who this affects.** Only members of the `realtime` group: you, unless you add other accounts. Other users of the PC keep their usual limits. The one shared risk: a buggy real-time program can hog the CPU. By default, Linux keeps 5 % of each second for normal processes (`/proc/sys/kernel/sched_rt_runtime_us` = `950000`), so the machine stays usable, but it can feel slow for everyone logged in while that happens.

**To undo:** remove the lines you added with `sudo nano /etc/security/limits.conf`, then `sudo gpasswd -d $USER realtime` and `sudo delgroup realtime`, and log out and in.

### 1.4 Lower latency on a normal kernel (Path B, recommended)

Your stock kernel is `PREEMPT_DYNAMIC`, which can be switched to its most responsive mode at boot. It's still not RT, but it makes late commands less frequent.

1. Open the boot config: `sudo nano /etc/default/grub`
2. Find the line `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"` and add `preempt=full` inside the quotes:
   ```
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash preempt=full"
   ```
3. Save (Ctrl+O, Enter) and exit (Ctrl+X). Then:
   ```bash
   sudo update-grub
   sudo reboot
   ```
4. Check: `sudo dmesg | grep -i "dynamic preempt"` should print `Dynamic Preempt: full`.

To undo, remove `preempt=full` and run `sudo update-grub` again.

### 1.5 CPU performance mode (both paths, every session)

Power saving makes the CPU slow to respond. Before every robot session:
```bash
powerprofilesctl set performance                                       # Ubuntu 22.04+/24.04
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor              # must print: performance
```
Both reset at reboot. On a **laptop**, also keep it **plugged into AC power**.

### 1.6 Network

Connect the PC's Ethernet port **directly** to the **LAN port of the FR3 Control box** (the controller, not the arm's base). Don't put a switch or router in between.

Give the PC a static IP on the robot's subnet. The example below uses `172.16.0.1` for the PC and port `enp2s0f0`. Replace both with your values (`ip -br link` lists ports):
```bash
sudo nmcli connection add type ethernet ifname enp2s0f0 con-name fr3 \
     ipv4.method manual ipv4.addresses 172.16.0.1/24 ipv6.method disabled
sudo nmcli connection up fr3
ping -c 5 $FR3_IP          # must reply, ~0.1–0.5 ms
```
If that port was also your wired office network, it's now used for the robot. Use Wi-Fi for internet in the meantime. To switch back later: `nmcli connection show` lists the connections, and `sudo nmcli connection up <office-connection-name>` reactivates the old one.

### 1.7 Build tools

```bash
sudo apt update
sudo apt install -y build-essential cmake git libpoco-dev libeigen3-dev
```
Newer libfranka versions (about 0.14 and later) need more packages, such as pinocchio and fmt. Check the README of your libfranka tag if the build in §3.2 complains about a missing package.

Install Miniconda if `conda --version` doesn't work.

### 1.8 Desktop session

The keyboard shortcuts in teleop (`t`, `n`, `u`/`o`, …) are read with `pynput`, which only sees key presses in an **X11** session. Check:
```bash
echo $XDG_SESSION_TYPE     # should print: x11
```
If it prints `wayland`, log out, click the gear icon on the login screen, choose **Ubuntu on Xorg**, and log in again.

**✅ Go/no-go 1:** `ping $FR3_IP` works, `ulimit -r` prints 99, and the CPU governor is `performance`. On Path A, `uname -a` shows `PREEMPT_RT`.

---

## 2. Prepare the robot in Desk

1. Power on the FR3 Control box and wait until the arm's lights stop blinking (about a minute).
2. In a browser on the PC, open `https://<FR3 IP>` (accept the certificate warning) and log in.
3. **System version:** if you haven't yet, note it under *Settings → System* (§1.1).
4. **End effector:** under *Settings → End-Effector*, select **Franka Hand**. Wrong settings here give wrong mass and gravity compensation.
5. **Unlock the joints:** click the unlock (padlock) button in the side panel. You'll hear the brakes click open.
6. **Activate FCI:** open the menu at the top right and choose **Activate FCI**. Leave the pop-up open, or note that it says FCI is active. Some FR3 versions also ask you to take "control token" ownership: accept it.
7. **Enabling device:** keep it in your hand from now on. Know how to stop the robot with it before any program moves the arm.

Desk must stay in this state (joints unlocked, FCI active) for every phase below. If the robot stops with an error, Desk shows it. Clear it there (or with the enabling device) before restarting the software.

**✅ Go/no-go 2:** Desk shows the joints unlocked and FCI active, with no errors.

---

## 3. Check that FCI works with your libfranka version

This builds a standalone libfranka **only for this check**. Polymetis builds its own copy in Phase 4.

### 3.1 Build

```bash
mkdir -p ~/fr3_test && cd ~/fr3_test
git clone --recursive https://github.com/frankaemika/libfranka --branch $LIBFRANKA_VER
cd libfranka
```

### 3.2 Patch the two examples (Path B only)

On a non-RT kernel, both examples stop with `Running kernel does not have realtime capabilities` unless patched. In older libfranka (0.10) only `communication_test` is affected, because the check runs when a control loop starts. In newer versions (0.18, probably earlier too) the check runs as soon as a `franka::Robot` is created, so even `echo_robot_state`, which only reads, is affected. Patching both works for every version:

```bash
# in ~/fr3_test/libfranka
for f in examples/communication_test.cpp examples/echo_robot_state.cpp; do
  sed -i 's/franka::Robot robot(argv\[1\]);/franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);/' $f
done
grep -n "franka::Robot robot" examples/communication_test.cpp examples/echo_robot_state.cpp
```
- `sed -i 's/OLD/NEW/' file` replaces the text OLD with NEW inside the file. `\[` and `\]` mean literal brackets.
- The `for` loop runs that on both files.
- `grep -n` prints the matching lines with their numbers, so you can check the result.

The `grep` must show `franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);` for **both** files. If a line looks different in your version, open the file (`nano examples/<file>.cpp`, Ctrl+W to search) and add `, franka::RealtimeConfig::kIgnore` after `argv[1]` by hand.

If you've already built (§3.3), rebuild after patching: `cd build && cmake --build . -j$(nproc)`. Any other example you want to run needs the same edit.

### 3.3 Build and run

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build . -j$(nproc)

./examples/echo_robot_state $FR3_IP         # prints robot state as text, then exits
./examples/communication_test $FR3_IP       # ~10 s; keep the enabling device in hand
```

What to look for:
- `echo_robot_state` prints blocks of numbers (joint positions `q`, etc.) and no error.
- `communication_test` prints a summary at the end with a success rate and the number of lost robot states. The success rate should be close to 1 (100 %) and lost states close to 0. On Path B, a few lost packets are normal. Many lost packets, or a `communication_constraints_violation` error, mean the timing is too poor.

| If you see | Do |
|---|---|
| `Incompatible library version (server version: X, library version: Y)` | Wrong libfranka for this FR3. Look up server version X in the compatibility table, pick a matching libfranka, set `LIBFRANKA_VER`, delete `~/fr3_test/libfranka` and redo §3 |
| `Running kernel does not have realtime capabilities` | Path B: that example isn't patched, or wasn't rebuilt after patching. Patch it (§3.2), then `cmake --build . -j$(nproc)` again |
| Many lost packets / `communication_constraints_violation` | Check §1.5 (performance mode, AC power), §1.6 (direct cable, built-in port), close heavy programs, try §1.4. If it stays bad, you need Path A |
| `Connection refused` / timeout | Wrong IP, cable in the arm's port instead of the Control box, or FCI not active (§2) |

**✅ Go/no-go 3:** both examples run without errors, and the communication test reports very few lost packets.

---

## 4. Build Polymetis against your libfranka

Use the official repo, which matches the `mamba install ... polymetis` setup in [src/real/setup.txt](../src/real/setup.txt). [furniture-bench/docker/server.Dockerfile](../furniture-bench/docker/server.Dockerfile) builds the same thing in Docker from a FurnitureBench fork, with a `LIBFRANKA_VERSION` build arg, if you prefer containers.

### 4.1 Create the env

```bash
cd ~/fr3_test
git clone --recursive https://github.com/facebookresearch/fairo
cd fairo/polymetis
conda env create -f ./polymetis/environment.yml     # creates the env "polymetis-local" (Python 3.8, torch 1.13); takes a while
conda activate polymetis-local
pip install -e ./polymetis
```

### 4.2 Build Polymetis' libfranka

```bash
# (polymetis-local), in ~/fr3_test/fairo/polymetis
./scripts/build_libfranka.sh $LIBFRANKA_VER
```
This checks out your version in Polymetis' own libfranka copy (`polymetis/src/clients/franka_panda_client/third_party/libfranka`) and builds it. If it says `pathspec '<version>' did not match`, fetch the tags first and rerun:
```bash
git -C polymetis/src/clients/franka_panda_client/third_party/libfranka fetch --tags
./scripts/build_libfranka.sh $LIBFRANKA_VER
```

**libfranka 0.14 or later: pinocchio version clash.** These versions need pinocchio. Inside the `polymetis-local` env, CMake finds the env's own **pinocchio 2.5.2** (not the `/opt/openrobots` one from §3), while libfranka is written for pinocchio 3. With 0.18.0, this gives one error:
```
src/robot_model.cpp:35:26: error: 'const pinocchio::Frame' ... has no member named 'parentJoint'
```
pinocchio 2.5 calls that field `parent`, and every other pinocchio function libfranka 0.18 uses exists in 2.5. Rename it and rerun the script:
```bash
# (polymetis-local), in ~/fr3_test/fairo/polymetis
sed -i 's/parent_frame\.parentJoint/parent_frame.parent/' \
    polymetis/src/clients/franka_panda_client/third_party/libfranka/src/robot_model.cpp
./scripts/build_libfranka.sh $LIBFRANKA_VER
```
Pass: it ends with `[100%] Built target franka`. Many `-Wmaybe-uninitialized` and `New Boost version may have incorrect or missing dependencies` warnings are normal.

Don't point this build at the `/opt/openrobots` pinocchio 3 instead: Polymetis itself links the env's pinocchio 2.5.2, and two pinocchio versions in one program can crash.

### 4.3 Patch for a non-RT kernel (Path B only)

```bash
# (polymetis-local), in ~/fr3_test/fairo/polymetis
F=polymetis/src/clients/franka_panda_client/franka_panda_client.cpp
sed -i 's/new franka::Robot(config\["robot_ip"\].as<std::string>())/new franka::Robot(config["robot_ip"].as<std::string>(), franka::RealtimeConfig::kIgnore)/' $F
grep -n "new franka::Robot" $F
```
The `grep` must show:
```cpp
robot_ptr_.reset(new franka::Robot(config["robot_ip"].as<std::string>(), franka::RealtimeConfig::kIgnore));
```
If it doesn't change, open the file, find `new franka::Robot(`, and add `, franka::RealtimeConfig::kIgnore` as the second argument by hand.

### 4.4 Build

**Newer libfranka and compilers (libfranka 0.18 with GCC 13 on Ubuntu 24.04): two source fixes first.** These were needed and sufficient on our setup. With older versions you may not need them, but they're harmless:
```bash
# (polymetis-local), in ~/fr3_test/fairo/polymetis/polymetis
# 1) libfranka 0.18's headers use std::optional, which needs C++17; Polymetis asks for C++14.
#    (Only the top-level file; torch_isolation/pinocchio_isolation keeps its own C++14 setting.)
sed -i '8s/-std=c++14/-std=c++17/' CMakeLists.txt
sed -n 8p CMakeLists.txt                       # must show: set(CMAKE_CXX_FLAGS "-std=c++17")

# 2) GCC 13 no longer pulls in size_t implicitly: "'size_t' has not been declared" in torch_server_ops.hpp
sed -i 's/^#include <map>$/#include <cstddef>\n#include <map>/' torch_isolation/include/torch_server_ops.hpp
grep -n "#include" torch_isolation/include/torch_server_ops.hpp   # must list <cstddef> first
```

Then configure and build:
```bash
# (polymetis-local), in ~/fr3_test/fairo/polymetis
mkdir -p ./polymetis/build && cd ./polymetis/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF \
      -Dfmt_DIR=$PWD/../src/clients/franka_panda_client/third_party/libfranka/build/_deps/fmt-build
make -j$(nproc)          # if the PC runs out of memory, use make -j4
ls | grep -E "run_server|franka_panda_client|franka_hand_client"
```
- `-Dfmt_DIR=…` is only needed for libfranka ≥ 0.14. Those versions need the fmt library; when fmt isn't installed, libfranka downloads and builds it inside its own build folder (`_deps/fmt-build`). This tells Polymetis where it is. Without it: `Could not find a package configuration file provided by "fmt"`.
- Warnings like `New Boost version may have incorrect or missing dependencies`, `kineto_LIBRARY-NOTFOUND` and `Cannot generate a safe runtime search path … libz.so.1` are harmless.

The last line should list all three programs. If you change the source later (for example, the §4.3 patch), just rerun `make -j$(nproc)` in this folder.

Optional check that the robot client uses the right libraries:
```bash
ldd franka_panda_client | grep -E "franka|pinocchio|not found"
# expect: libfranka.so.<your version> from …/third_party/libfranka/build, libpinocchio.so.2.5.2 from the env, and no "not found"
```

| If you see | Do |
|---|---|
| Errors inside `franka_panda_client*.cpp` | libfranka's API changed. Read the error. Small fixes (a renamed header or function, a new argument) are fine to patch. If it needs real porting, that's a no-go for this route |
| Errors inside libfranka, like `'uint8_t' does not name a type` | Older libfranka with a newer compiler. Add `#include <cstdint>` at the top of the header the error names, then rebuild (§4.2, then §4.4) |
| libfranka can't find a package | Newer libfranka needs extra dependencies. Install what its README lists for your tag |
| `'optional' in namespace 'std' does not name a template type` | New libfranka headers need C++17 | Fix 1 above |
| `'size_t' has not been declared` in `torch_server_ops.hpp` | New compiler | Fix 2 above |
| `Could not find … "fmt"` during `cmake` | libfranka ≥ 0.14's fmt isn't installed system-wide | Add the `-Dfmt_DIR=…` option above |

### 4.5 Fix the stop command (required on Ubuntu 24.04, both paths)

⚠️ **Do this before the first `launch_robot.py`.** Without it, pressing Ctrl+C in the robot server **kills every program in your desktop session** (terminal, browser, editor, the desktop itself), which looks like the PC shutting down. On Path A, where the cleanup runs with `sudo`, it kills every process on the whole system.

Why: on Ctrl+C, `launch_robot.py` stops its server with `kill -9 -<pgid>` ("kill process group `<pgid>`"). Ubuntu 24.04's `kill` (procps 4.0.4) misreads a negative number given like that and keeps only its first digit. For example, `kill -9 -111629` becomes `kill -9 -1`, and `-1` means "every process I'm allowed to signal". Adding `--` before the number makes it parse correctly.

```bash
# (polymetis-local), in ~/fr3_test/fairo/polymetis/polymetis
sed -i 's/\["sudo", "kill", "-9", f"-{pgid}"\]/["sudo", "kill", "-9", "--", f"-{pgid}"]/; s/\["kill", "-9", f"-{pgid}"\]/["kill", "-9", "--", f"-{pgid}"]/' \
    python/scripts/launch_robot.py
grep -n '"kill"' python/scripts/launch_robot.py
```
Both lines the `grep` prints must contain `"--"`:
```
63:            subprocess.check_call(["sudo", "kill", "-9", "--", f"-{pgid}"])
69:            subprocess.check_call(["kill", "-9", "--", f"-{pgid}"])
```
No rebuild is needed: it's a Python script, and the editable install (§4.1) uses the file directly.

You can check whether your `kill` has the bug, harmlessly (signal 0 doesn't kill anything):
```bash
strace -e trace=kill /usr/bin/kill -0 -45678 2>&1 | grep "^kill("
# kill(-4, 0)     → bug present (only the first digit is kept)
# kill(-45678, 0) → no bug
```
(`sudo apt install strace` if it's missing.) Apply the fix either way: with `--`, the command is correct for every version of `kill`.

### 4.6 Fix the torque-control start (required with libfranka 0.18 on the FR3)

**Symptom without it:** `launch_robot.py` prints `Connected.` and then nothing; the arm isn't held, and any script fails with `Robot context not valid when calling GetRobotClientMetadata!`. No error appears anywhere.

**Cause** (found with a stack trace and three small test programs, `~/fr3_test/control_loop_test.cpp`): with libfranka 0.18 on our FR3, starting torque control as the **first motion after connecting** never gets past libfranka's start sequence. The robot answers "motion started" before its state stream shows the new mode. libfranka then decides the motion was aborted and waits forever for an abort message that never comes. It happens with both of libfranka's torque APIs. A short motion beforehand avoids it: after 0.5 s of "hold still" (zero joint velocity), torque control starts normally. `communication_test` never hits it, because it moves to a start pose first.

**Fix:** make the robot client hold still for 0.5 s before each torque-control start, including restarts after automatic error recovery. Open `polymetis/src/clients/franka_panda_client/franka_panda_client.cpp`, find `robot_ptr_->control(control_callback, limit_rate_, lpf_cutoff_freq_);` in `run()`, and insert the block below **directly above** it, inside the same `try {`:
```cpp
        // FR3 + libfranka 0.18: starting torque control as the first motion after connecting (or
        // after error recovery) deadlocks in libfranka's start sequence. A short zero-velocity
        // motion first avoids it; the arm does not move.
        int hold_cycles = 0;
        robot_ptr_->control(
            [&](const franka::RobotState &, franka::Duration) -> franka::JointVelocities {
              franka::JointVelocities zero_velocities{{0, 0, 0, 0, 0, 0, 0}};
              if (++hold_cycles >= 500) {
                return franka::MotionFinished(zero_velocities);
              }
              return zero_velocities;
            });
        spdlog::info("Hold-still motion done; starting torque control.");
```
Then rebuild the robot client:
```bash
# (polymetis-local), in ~/fr3_test/fairo/polymetis/polymetis/build
make -j$(nproc) franka_panda_client
```
With the fix, terminal 1 prints `Hold-still motion done; starting torque control.` right after `Connected.`.

**✅ Go/no-go 4:** `make` finishes, `run_server`, `franka_panda_client` and `franka_hand_client` exist in `polymetis/build`, `launch_robot.py` has the `--` fix (§4.5), and the robot client has the hold-still fix (§4.6).

---

## 5. Polymetis server on the FR3

### 5.1 Set the FR3 joint limits

Polymetis refuses commands outside its joint limits, and its defaults are the Panda's. Open:
```bash
nano ~/fr3_test/fairo/polymetis/polymetis/conf/robot_client/franka_hardware.yaml
```
Find `joint_pos_upper:` and `joint_pos_lower:` under `limits:`, and replace their seven values with these. They're the FR3 datasheet limits minus the same 0.1 rad margin Polymetis uses for the Panda:
```yaml
      joint_pos_upper: #FR3, margin: 0.1 rad
        - 2.64
        - 1.68
        - 2.80
        - -0.26
        - 2.70
        - 4.41
        - 2.91
      joint_pos_lower: #FR3, margin: 0.1 rad
        - -2.64
        - -1.68
        - -2.80
        - -2.94
        - -2.70
        - 0.65
        - -2.91
```
Keep the indentation exactly as in the file (spaces, not tabs). The FR3 limits these come from:

| Joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| FR3 min (rad) | -2.7437 | -1.7837 | -2.9007 | -3.0421 | -2.8065 | **0.5445** | -3.0159 |
| FR3 max (rad) | 2.7437 | 1.7837 | 2.9007 | -0.1518 | 2.8065 | **4.5169** | 3.0159 |

Check them against your robot's datasheet. The teleop home pose `[-0.093, 0.038, 0.149, -2.381, -0.009, 2.495, 0.120]`, plus ±5° of noise, falls well inside.

Leave everything else as it is:
- `joint_vel` and `joint_torques` are already below the FR3's limits.
- The Cartesian safety box (`cartesian_pos_upper/lower`, in metres from the robot base) is fine for free-space tests. If the arm later stops near the table without an error, widen it.
- Leave `robot_ip` alone. You pass it on the command line.

### 5.2 Launch the servers

Two terminals, both in `polymetis-local`. These are the same commands as [furniture-bench/launch_daemon.sh](../furniture-bench/launch_daemon.sh), with the FR3's IP.

Path A (RT kernel):
```bash
# terminal 1
conda activate polymetis-local
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=$FR3_IP
```
It asks for your `sudo` password, because it runs the server with real-time priority.

Path B (no RT kernel):
```bash
# terminal 1
conda activate polymetis-local
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=$FR3_IP use_real_time=false
```
No `sudo`. The control thread still gets high priority through the permissions from §1.3.

Both paths:
```bash
# terminal 2
conda activate polymetis-local
launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=$FR3_IP
```

What should happen: terminal 1 logs that it connected to the robot and keeps running. The arm stays where it is and feels slightly springy if you push it gently: Polymetis holds it with a soft default controller. Terminal 2 connects to the gripper and keeps running.

To stop the servers: Ctrl+C in each terminal. This is only safe with the §4.5 fix in place. If a later launch says `Port unavailable; possibly another server found`, an old server is still running: `pkill -9 run_server` (on Path A: `sudo pkill -9 run_server`).

**✅ Go/no-go 5:** both servers keep running with no `Incompatible library version`, `realtime` or reflex errors, and the arm holds its pose.

### 5.3 Polymetis-only sanity checks

Save this as `~/fr3_test/poly_check.py`. The arm moves joint 7 by about 6° and back, and the gripper opens and closes: keep the enabling device in hand and nothing between the fingers.
```python
import time, torch
from polymetis import RobotInterface, GripperInterface

robot = RobotInterface(ip_address="localhost")
gripper = GripperInterface(ip_address="localhost")
print("q   =", robot.get_joint_positions())
print("ee  =", robot.get_ee_pose())             # flange pos, quat xyzw — compare with Desk
print("grip=", gripper.get_state().width)

# 1) small joint move: +0.1 rad on joint 7 and back
q0 = robot.get_joint_positions()
robot.move_to_joint_positions(q0 + torch.tensor([0, 0, 0, 0, 0, 0, 0.1]))
robot.move_to_joint_positions(q0)

# 2) gripper: open, then close until contact (Polymetis: grasp(speed, force))
gripper.goto(0.08, 0.05, 0.1, blocking=True)
gripper.grasp(0.07, 70, blocking=True)
gripper.goto(0.065, 0.05, 0.1, blocking=True)
```
Run it while both servers are running:
```bash
# terminal 3
conda activate polymetis-local
python ~/fr3_test/poly_check.py
```

Pass:
- `q` matches the joint angles Desk shows (Desk shows degrees; `q` is in radians).
- `ee` position is close to Desk's. If Desk shows the Franka Hand's fingertip point (TCP) instead of the flange, the two differ by about 0.10 m along the gripper's axis. That's expected.
- The joint moves smoothly there and back.
- The gripper opens, closes fully, then opens to 6.5 cm.

---

## 6. Install `rdt` (the authors' tools)

⚠️ **Do not `pip install rdt`.** That installs an unrelated PyPI package. [src/real/setup.txt](../src/real/setup.txt) says `pip install rdt`, but that's wrong for this purpose.

```bash
# (polymetis-local)
cd ~/fr3_test
pip uninstall -y rdt                          # in case the PyPI one is installed; "not installed" is fine
git clone -b lars-dev https://github.com/anthonysimeonov/improbable_rdt.git
pip install cython                            # setup.py imports Cython on its first lines (it compiles nothing)
pip install -e improbable_rdt                 # package lives in src/rdt; setup.py declares no deps
```
- No `--recurse-submodules`: `lars-dev` has two submodules (`pytorch_contactnet`, `pybullet-planning`), but none of the modules teleop imports uses them. Their URLs are SSH (`git@github.com:…`), so fetching them would also fail without a GitHub SSH key.
- Without Cython, `pip install -e` fails with `ModuleNotFoundError: No module named 'Cython'`.

`rdt` finds its own files (message definitions in `lcm_types`, configs) through the environment variable `RDT_SOURCE_DIR`. Without it, importing `rdt.image.factory` fails with `KeyError: 'RDT_SOURCE_DIR'`. The repo's `rdt_env.sh` sets it, but only when you `source` it by hand. Store it in the conda env instead, so it's set on every `conda activate polymetis-local`:
```bash
conda env config vars set -n polymetis-local RDT_SOURCE_DIR=$HOME/fr3_test/improbable_rdt/src/rdt
conda deactivate && conda activate polymetis-local     # once, so it takes effect
echo $RDT_SOURCE_DIR                                    # → /home/<you>/fr3_test/improbable_rdt/src/rdt
```
(`rdt_env.sh` also sets two variables for the submodules; teleop doesn't need them.)

**pip and company package servers.** If your `pip install`s hang with `Retrying … ConnectTimeoutError` for an internal server (set in `/etc/pip.conf`), point this env at public PyPI: `pip config --site set global.index-url https://pypi.org/simple`. It's undone with `pip config --site unset global.index-url`.
`lars-dev` is the branch with every module `teleop_sm.py` and `minimal.py` import:
- `spacemouse/spacemouse_shared_memory.py`
- `polymetis_robot_utils/{polymetis_util, interfaces/diffik}.py`
- `common/{mc_util, keyboard_interface, demo_util}.py`
- `image/factory.py`
- `config/default_multi_realsense_cfg.py`

Its RealSense helper streams **rgb8**, which matches the sim. Its keyboard maps `t` to success, `n` to fail, `p` to pause (rollout only), and `i/k/j/l/u/o` to rotation.

Other dependencies, in the same env (from [src/real/setup.txt](../src/real/setup.txt) and [src/real/readme.md](../src/real/readme.md)):
```bash
# (polymetis-local)
pip install pyrealsense2 meshcat pynput yacs lcm ipdb opencv-python termcolor atomics scipy

# SpaceMouse driver (system-wide) and its Python binding
sudo apt install -y libspnav-dev spacenavd
sudo systemctl enable --now spacenavd
pip install git+https://github.com/cheng-chi/spnav
```
Unplug and replug the SpaceMouse afterwards.

`pyrealsense2` is needed even without cameras: `teleop_sm.py` and `rdt` import it at startup. It installs fine with no camera attached.

**✅ Go/no-go 6: all teleop imports work.**
```bash
# (polymetis-local)
python - <<'PY'
from polymetis import GripperInterface
from rdt.spacemouse.spacemouse_shared_memory import Spacemouse
from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.common.demo_util import CollectEnum
from rdt.image.factory import enable_single_realsense
import inspect; print(inspect.signature(DiffIKWrapper.__init__))
PY
```
Pass: no errors, and the last line prints `(self, robot_home, Kq, Kqd, *args, **kwargs)`. That signature is why Phase 8 changes line 389. `rdt` doesn't declare its dependencies, so if an import fails with `No module named 'X'`, run `pip install X` and try again.

---

## 7. SpaceMouse on its own

```bash
# (polymetis-local)
python - <<'PY'
import time
from multiprocessing.managers import SharedMemoryManager
from rdt.spacemouse.spacemouse_shared_memory import Spacemouse
with SharedMemoryManager() as shm, Spacemouse(shm_manager=shm, deadzone=0.05) as sm:
    for _ in range(100):
        print(sm.get_motion_state_transformed().round(2), sm.is_button_pressed(0), sm.is_button_pressed(1))
        time.sleep(0.1)
PY
```
For 10 seconds, push, pull, twist and tilt the cap, and press both buttons.

Pass: the six numbers change while you move the cap, and return to 0 when you let go. The two `True/False` values switch when you press the buttons. If everything stays 0: `systemctl status spacenavd` must say `active (running)`. If not, `sudo systemctl restart spacenavd` and replug.

**✅ Go/no-go 7:** the SpaceMouse responds.

---

## 8. Minimal edits to `teleop_sm.py`

There are two ways to get a teleop script that runs on the FR3:
- **Use the ready-made [src/real/teleop_sm_fr3.py](../src/real/teleop_sm_fr3.py)** (§8.2, recommended). It's a copy of the original with the six edits below already made, plus a `--no_gripper` switch for running without a gripper. The original `teleop_sm.py` stays untouched.
- **Edit `teleop_sm.py` by hand** (§8.1). §8.1 explains each change; read it either way, since `teleop_sm_fr3.py` contains exactly these edits.

### 8.1 The edits, one by one

Only what's needed to start it, plus a switch to run without cameras. Open [src/real/teleop_sm.py](../src/real/teleop_sm.py) and make these six changes.

The line numbers are from the unedited file. Change 4 adds a line, so everything below it shifts down. Use Ctrl+F with the quoted code to find each spot.

**1. Line [24](../src/real/teleop_sm.py#L24):** comment out the unused FurnitureBench import. Its only use is commented out, and it would pull `furniture_bench` and `gym` into the Polymetis env.
```python
# from furniture_bench.utils.scripted_demo_mod import scale_scripted_action
```

**2. Line [387](../src/real/teleop_sm.py#L387):** point it at the local Polymetis server. This is the **server** address, not the robot's IP, and the server runs on this PC.
```python
    franka_ip = "localhost"
```

**3. Line [389](../src/real/teleop_sm.py#L389):** match the `lars-dev` constructor (go/no-go 6). It stores these three values only for its own `reset()`, which teleop doesn't call.
```python
    robot = DiffIKWrapper(robot_home=None, Kq=None, Kqd=None, ip_address=franka_ip)
```

Changes 4–6 add the `--no_cameras` switch. With it, the script doesn't open any camera and records **black images of the same size and type** (640×480 RGB, `uint8`) in place of the camera frames. The saved episodes keep the exact format of real ones, so Phase 10 can check the whole data pipeline. Without the switch, the script behaves exactly as before.

**4. After line [41](../src/real/teleop_sm.py#L41)** (`parser.add_argument("--furniture", ...)`), add the option:
```python
parser.add_argument("--furniture", type=str, required=True)
parser.add_argument("--no_cameras", action="store_true")
```

**5. Line [304](../src/real/teleop_sm.py#L304)** (`def get_rgbd_rs(self, pipe) -> dict:`): return a black image when there's no camera. Add the four lines below right after the `def` line, before `align_to = rs.stream.color`:
```python
    def get_rgbd_rs(self, pipe) -> dict:
        if pipe is None:  # --no_cameras: black images in the same format as the D435s
            return dict(
                rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                depth=np.zeros((480, 640), dtype=np.uint16),
            )
        align_to = rs.stream.color
```

**6. Lines [418–436](../src/real/teleop_sm.py#L418-L436)** (from `# Setup camera streams` down to the second `time.sleep(1.0)`): replace the whole block with this. It's the same code, indented under an `else:`, with a camera-free branch in front:
```python
    # Setup camera streams
    if args.no_cameras:
        print("--no_cameras: not opening any camera, saving black placeholder images")
        front_image_pipeline = wrist_image_pipeline = None
    else:
        rs_cfg = get_default_multi_realsense_cfg()
        resolution_width = rs_cfg.WIDTH  # pixels
        resolution_height = rs_cfg.HEIGHT  # pixels
        frame_rate = rs_cfg.FRAME_RATE  # fps

        ctx = rs.context()  # Create librealsense context for managing devices

        print(
            f"Enabling devices with serial numbers: {FRONT_CAMERA_SERIAL} (front) and {WRIST_CAMERA_SERIAL} (wrist)"
        )
        front_image_pipeline = enable_single_realsense(
            FRONT_CAMERA_SERIAL, ctx, resolution_width, resolution_height, frame_rate
        )
        time.sleep(1.0)
        wrist_image_pipeline = enable_single_realsense(
            WRIST_CAMERA_SERIAL, ctx, resolution_width, resolution_height, frame_rate
        )
        time.sleep(1.0)
```

Check that the file still parses, then review the edits:
```bash
# (polymetis-local), in ~/ResiP
python -m py_compile src/real/teleop_sm.py && echo OK
git diff src/real/teleop_sm.py
```
`OK` means no syntax or indentation mistakes. To undo all six changes: `git checkout src/real/teleop_sm.py`.

### 8.2 The ready-made version: `teleop_sm_fr3.py`

[src/real/teleop_sm_fr3.py](../src/real/teleop_sm_fr3.py) is `teleop_sm.py` with changes 1–6 from §8.1 already made, in the same places, plus these extras:

| Extra | What it does |
|---|---|
| `--server_ip` option, default `localhost` | Change 2 as an option instead of a fixed value. Only needed if the Polymetis server runs on another PC |
| `--no_gripper` option | Starts without a gripper server. A stand-in gripper (class `NoGripper`) accepts the open/close commands and reports the commanded width as `gripper_width`: 6.5 cm when open, 0 after "close" (the Franka Hand closes fully when there's nothing to grasp) |

To see every difference from the original:
```bash
git diff --no-index src/real/teleop_sm.py src/real/teleop_sm_fr3.py
```

With `--no_cameras` and `--no_gripper`, the saved episodes keep the exact format of real ones, so Phase 10 can still check the whole data pipeline. Both choices are saved in each episode's `args`, so placeholder episodes can always be identified later. Without the options, the script behaves like the hand-edited version.

Check that it loads in your env. `--help` exits before touching the robot:
```bash
# (polymetis-local), in ~/ResiP
python -m src.real.teleop_sm_fr3 --help
```
The option list must end with `[--server_ip SERVER_IP] [--no_cameras] [--no_gripper]`. An import error here means a §6 step is missing.

**Running without a gripper (`--no_gripper`):**
- Keep Desk's end effector set to **None** (*Settings → End-Effector*). The robot uses this setting to compensate the hand's weight; a wrong setting makes the arm drift.
- Don't start the gripper server (terminal 2 in §9.1). Without a hand, it stops with `Connection to FCI refused`, which is expected.
- In §9.2, check d (SpaceMouse button) only changes the recorded gripper command and width; nothing moves on the robot.
- When you mount the Franka Hand later: set the end effector back to Franka Hand in Desk, start the gripper server, and drop `--no_gripper`.

---

## 9. First teleop run, in free space (no cameras)

No parts on the table yet. Keep the enabling device in hand the whole time.

Before the first run, read [FR3_POLYMETIS_SAFETY_REVIEW.md](FR3_POLYMETIS_SAFETY_REVIEW.md): it lists what can make the robot move faster or less expectedly than you'd assume (automatic restart after a stop, global keyboard, real speeds, diff-IK near singularities and limits, collision thresholds) and the planned fixes.

⚠️ **As soon as the script starts, the arm drives to the home pose** (gripper pointing down, in front of the base, with up to ±5° of random noise per joint). Make sure that path is clear before starting it.

### 9.1 Start everything

You need four terminals:
```bash
# terminal 1: robot server, as in §5.2 (with use_real_time=false on Path B)
# terminal 2: gripper server, as in §5.2 (skip it with no gripper mounted, see §8.2)

# terminal 3: visualizer
conda activate polymetis-local
meshcat-server
```
`meshcat-server` prints a `zmq_url` (port `6000` by default) and a `web_url` (like `http://127.0.0.1:7000/static/`). Open the `web_url` in a browser to see the target and current gripper frames.

```bash
# terminal 4: teleop, from the repo root
conda activate polymetis-local
cd ~/ResiP
python -m src.real.teleop_sm -p 6000 --furniture one_leg --save_dir scratch/fr3_teleop_nocam \
       --max_pos_speed 0.15 --max_rot_speed 0.4 --no_cameras
```
With the ready-made script (§8.2), use this instead. Add `--no_gripper` if no gripper is mounted:
```bash
# terminal 4: teleop with teleop_sm_fr3.py, from the repo root
conda activate polymetis-local
cd ~/ResiP
python -m src.real.teleop_sm_fr3 -p 6000 --furniture one_leg --save_dir scratch/fr3_teleop_nocam \
       --max_pos_speed 0.15 --max_rot_speed 0.4 --no_cameras --no_gripper
```
- `-p` must match meshcat's zmq port.
- The two speed options halve the defaults (0.3 m/s and 0.7 rad/s) for this first session. Drop them once you're comfortable.
- `--no_cameras` is the switch from §8. The `nocam` folder name keeps these placeholder episodes apart from real ones later.
- `--no_gripper` (only in `teleop_sm_fr3.py`) replaces the gripper server with a stand-in (§8.2).

What happens in order: the gripper opens to 6.5 cm (with `--no_gripper`, it prints `--no_gripper: …` instead), the arm moves to the home pose, joint impedance starts, it prints `--no_cameras: not opening any camera, ...`, and then `Start collecting!`. From here, the SpaceMouse moves the arm.

### 9.2 Check, in order

| # | Do | Expect |
|---|---|---|
| a | Nothing | Arm holds still; no faults |
| b | Push the SpaceMouse gently forward / left / up | Arm moves in a consistent direction (the mapping assumes you face the robot from the front, where the front camera will go); meshcat target and current frames move together |
| c | Twist and tilt | Gripper rotates; no jumps |
| d | SpaceMouse button | Gripper closes; press again → opens |
| e | Keyboard `u` / `o` (focus anywhere in the desktop session) | Rotation about the gripper axis, as used for screwing |
| f | Move near the table and joint limits slowly | Server either follows or stops cleanly. No reflex error loops |
| g | Press `t` | Loop ends and a `<timestamp>.pkl` appears in `scratch/fr3_teleop_nocam/` |
| h | Relaunch terminal 4, move, press `n` | Loop ends, nothing saved |

### 9.3 If the robot stops

1. Look at terminal 1 and Desk for the error. On Path B, `communication_constraints_violation` is the usual one (§0.3).
2. Ctrl+C in terminal 4, then in terminals 1 and 2.
3. Clear the error in Desk (or with the enabling device) so the robot is ready again (§2).
4. Restart terminals 1 and 2 (§5.2), then terminal 4. Terminal 3 can keep running.

Occasional stops on Path B are expected. If they happen every few seconds, recheck §1.5 and §1.6, close other programs, and consider Path A.

**✅ Go/no-go 8:** a–h all behave as expected.

### 9.4 Shutting down

Ctrl+C in terminals 4, 3, 2 and 1, in that order. In Desk, deactivate FCI and lock the joints before powering off.

---

## 10. Check the recorded data (no cameras)

### 10.1 Contents

In the **main ResiP env** (`resip`), not the Polymetis one:
```bash
# (resip), in ~/ResiP
python - <<'PY'
import glob, pickle
p = sorted(glob.glob("scratch/fr3_teleop_nocam/*.pkl"))[-1]
d = pickle.load(open(p, "rb"))
print(d.keys(), len(d["observations"]), len(d["actions"]))
o = d["observations"][0]
print(o["color_image1"].shape, o["color_image1"].dtype, "max pixel:", o["color_image1"].max())
print(o["robot_state"].keys())
print("action[0] =", d["actions"][0])          # [dx dy dz qx qy qz qw grasp]
PY
```
Expect:
- Images `(480, 640, 3) uint8` with `max pixel: 0`, the black placeholders.
- `robot_state` with `ee_pos, ee_quat, ee_pos_vel, ee_ori_vel, gripper_width`.
- The same number of observations and actions.
- Small deltas (mm scale) in each action.

### 10.2 Through the pipeline

1. Copy the pickles into the raw data tree under a separate name, `one_leg_fr3test_nocam`, so they can never be mixed with real training data (needs `DATA_DIR_RAW` set, see [DATA_GUIDE.md](DATA_GUIDE.md) §1):
   ```bash
   D=$DATA_DIR_RAW/raw/diffik/real/one_leg_fr3test_nocam/teleop/low/success
   mkdir -p $D && cp scratch/fr3_teleop_nocam/*.pkl $D/
   ```
2. Convert them to a zarr:
   ```bash
   python -m src.data_processing.process_pickles -c diffik -d real -f one_leg_fr3test_nocam -s teleop -r low -o success
   ```
3. Check the robot data with the viewers ([DATA_GUIDE.md](DATA_GUIDE.md) §9.2). Skip `video`: the images are black.
   ```bash
   python scripts/visualize_data.py info $D
   python scripts/visualize_data.py plot $D
   ```
   `info` must report no NaNs. In the plots in `viz/`, the end-effector path should follow the commanded target closely, and the gripper width should follow the open/close commands.

**✅ Go/no-go 9:** the pickle has the right content, processes into a zarr, and the plots look right. **Stage 1 done: the original teleop code drives your FR3.** Continue with Phase 11 when you have the cameras. Don't use the `nocam` data for training.

---

## 11. Add the cameras

Everything from Phases 1–10 stays as it is. You only add the cameras, then repeat the teleop run without `--no_cameras`.

### 11.1 USB permissions

pip's `pyrealsense2` doesn't install the USB rules that let a normal user open the cameras. Install them once:
```bash
sudo curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
     -o /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 11.2 Check that both cameras are seen

Plug both D435s into **USB 3** ports (often blue, or marked SS). If they were plugged in before §11.1, unplug and replug them.
```bash
# (polymetis-local)
python -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.serial_number), 'USB', d.get_info(rs.camera_info.usb_type_descriptor)) for d in rs.context().devices]"
```
Pass: two lines, each with a serial number and `USB 3.x`. `USB 2.x` means the wrong port or cable, and the camera can't stream properly on USB 2. No lines at all means the rules from §11.1 aren't active yet (replug), or a cable problem.

### 11.3 Mount

Mount both cameras rigidly in their final positions now: one on the wrist, one at the front. See REAL_ROBOT_FR3_GUIDE.md §5. Route the wrist camera's cable so it can't catch on the arm when the arm moves to the home pose or rotates the gripper.

### 11.4 Tell them apart and enter the serials

This saves one picture from each camera, named after its serial:
```bash
# (polymetis-local), in ~/fr3_test
python - <<'PY'
import numpy as np, cv2, pyrealsense2 as rs
for d in rs.context().devices:
    s = d.get_info(rs.camera_info.serial_number)
    p, c = rs.pipeline(), rs.config()
    c.enable_device(s)
    c.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    p.start(c)
    for _ in range(30):                      # let auto-exposure settle
        f = p.wait_for_frames()
    img = np.asanyarray(f.get_color_frame().get_data())
    cv2.imwrite(f"cam_{s}.png", img[:, :, ::-1])   # RGB → BGR for OpenCV
    p.stop()
    print("saved", f"cam_{s}.png")
PY
```
Open the two images in `~/fr3_test`. The one that shows the gripper fingers up close is the wrist camera.

Write the serials into [src/real/serials.py](../src/real/serials.py), replacing the authors' values:
```python
FRONT_CAMERA_SERIAL = "<serial of the front camera>"
WRIST_CAMERA_SERIAL = "<serial of the wrist camera>"
```
Swapping them silently breaks the data, so double-check against the images.

**✅ Go/no-go 10:** both cameras show up on USB 3, and `serials.py` has the right serial for each.

### 11.5 Teleop with cameras

Start terminals 1–3 as in §9.1. Then, in terminal 4, run the same command **without `--no_cameras`**, into a new folder:
```bash
# terminal 4
conda activate polymetis-local
cd ~/ResiP
python -m src.real.teleop_sm -p 6000 --furniture one_leg --save_dir scratch/fr3_teleop_test \
       --max_pos_speed 0.15 --max_rot_speed 0.4
```
With `teleop_sm_fr3.py`: the same command with `src.real.teleop_sm_fr3`, keeping `--no_gripper` if there's still no gripper mounted.

What changes compared with §9: after the arm reaches home, it prints `Enabling devices with serial numbers: ... (front) and ... (wrist)` and waits about 2 s for the cameras before `Start collecting!`.

Repeat checks a, b, g and h from §9.2. Move around for at least 30 s before pressing `t`, so you can see that the cameras keep up: the loop must not stop with `Couldn't get frame for device`.

### 11.6 Check the recorded data

1. Contents: run the script from §10.1 with `fr3_teleop_nocam` replaced by `fr3_teleop_test`. Now `max pixel` must be well above 0 (usually 200–255).
2. Pipeline: as in §10.2, with the real dataset name `one_leg_fr3test`:
   ```bash
   D=$DATA_DIR_RAW/raw/diffik/real/one_leg_fr3test/teleop/low/success
   mkdir -p $D && cp scratch/fr3_teleop_test/*.pkl $D/
   python -m src.data_processing.process_pickles -c diffik -d real -f one_leg_fr3test -s teleop -r low -o success
   python scripts/visualize_data.py video $D --fps 10
   python scripts/visualize_data.py plot  $D
   ```
3. In the video in `viz/`, check:
   - The left panel is the wrist camera (`color_image1`) and the right one the front camera. If they're swapped, swap the serials in `serials.py`.
   - The colors look natural. Blue skin or a blue-tinted orange means red and blue are swapped.
   - The motion is smooth, with no frozen or skipped stretches.

**✅ Go/no-go 11:** the pickle has real images, processes into a zarr, and the video looks right. **At this point the original code works on your FR3 for teleop, with cameras.** Next: build the scene (REAL_ROBOT_FR3_GUIDE.md §5.2) and collect real demos (§6 there).

---

## 12. Every session after setup

Once everything above passes, a session is:

1. Power on the FR3. In Desk: unlock joints, activate FCI (§2).
2. On the PC: performance mode (§1.5); laptop on AC power.
3. `nmcli connection up fr3` if the Ethernet port was used for something else; `ping $FR3_IP`.
4. Terminals 1–4 as in §9.1. Add `--no_cameras` only if the cameras aren't connected.
5. Collect. Afterwards, shut down as in §9.4.

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `libfranka: Running kernel does not have realtime capabilities.` | Non-RT kernel, and libfranka set to enforce RT | Path B patches: §3.2 (examples), §4.3 (Polymetis, then rebuild §4.4) |
| `unable to set realtime scheduling` / `Operation not permitted` | No RT permissions | §1.3, then log out and in; `ulimit -r` must print 99 |
| `libfranka: Incompatible library version (server version: X, library version: Y)` | libfranka doesn't match the FR3 system version | Pick a matching version (§1.1), rebuild §3 and §4 |
| `communication_constraints_violation`, lost packets | Timing too poor: no RT kernel, power saving, USB-Ethernet adapter, a switch in the path, or heavy load | §1.4–1.6; close other programs; if still frequent, Path A (§1.2) |
| `pathspec '<version>' did not match` in `build_libfranka.sh` | Submodule has no tags yet | `git -C …/third_party/libfranka fetch --tags`, rerun (§4.2) |
| Polymetis build errors in `franka_panda_client` | libfranka API change | Patch small renames. If it needs real porting → switch to REAL_ROBOT_FR3_GUIDE.md |
| `'uint8_t' does not name a type` (or similar) in libfranka | Old libfranka, new compiler | Add `#include <cstdint>` to that header, rebuild |
| Hydra: `Could not override 'gripper.executable_cfg.robot_ip'` (or another key) | Key name differs in your Polymetis checkout | Check the keys in `conf/robot_client/franka_hardware.yaml` and `conf/gripper/franka_hand.yaml` |
| Ctrl+C in the robot server closes all your programs / logs you out / looks like the PC shut down | `kill -9 -<pgid>` misparsed as `kill -9 -1` (Ubuntu 24.04 procps bug) | §4.5. Don't run `launch_robot.py` without it: every way of exiting runs the same cleanup |
| Server prints `Connected.` then nothing; scripts fail with `Robot context not valid when calling GetRobotClientMetadata!` | Torque control can't start as the first motion (libfranka 0.18 + FR3) | §4.6 |
| `Port unavailable; possibly another server found` | Old server still running | `pkill -9 run_server` (Path A: `sudo pkill -9 run_server`) |
| `ModuleNotFoundError: rdt.spacemouse` (or similar) | Wrong `rdt` (PyPI or `master`) | `pip uninstall rdt`, then install the `lars-dev` clone (§6) |
| `KeyError: 'RDT_SOURCE_DIR'` | `rdt`'s source-folder variable not set | §6: `conda env config vars set …`, then reactivate the env |
| `ModuleNotFoundError: pyrealsense2` | Not installed, even though there are no cameras | `pip install pyrealsense2` (§6); the script imports it either way |
| `TypeError: __init__() missing ... 'robot_home', 'Kq', 'Kqd'` | `lars-dev` `DiffIKWrapper` signature | §8, change 3 |
| `ModuleNotFoundError: furniture_bench` | Line 24 not commented out | §8, change 1 |
| `unrecognized arguments: --no_cameras` | §8 changes 4–6 not made | Make them; check with `python -m py_compile` |
| Teleop hangs or errors at startup with no gripper mounted (gripper `RpcError` / `UNAVAILABLE` on port 50052) | `GripperInterface` has no gripper server to talk to | Use `teleop_sm_fr3.py` with `--no_gripper` (§8.2) |
| `IndentationError` / `SyntaxError` in `teleop_sm.py` | Change 5 or 6 pasted with the wrong indentation | Compare with §8; method body lines start with 8 spaces, the `main()` block with 4 |
| Camera error at startup (`No device connected`, `Couldn't resolve requests`, …) with no cameras attached | `--no_cameras` missing | Add `--no_cameras` to the command |
| Client/server version mismatch error | `polymetis` client and server from different builds | Use the one env from §4 for both. `RobotInterface` also accepts `enforce_version=False` |
| Reflex errors: joint limits, velocity or torque discontinuity | Commands outside FR3 limits, or targets too aggressive | FR3 limits (§5.1); lower `--max_pos_speed` / `--max_rot_speed`; recover in Desk |
| Arm stops near the table with no reflex | Polymetis Cartesian safety box | Widen `cartesian_pos_lower/upper` in `franka_hardware.yaml` |
| SpaceMouse values stay 0 | `spacenavd` not running, or device not seen | `sudo systemctl restart spacenavd`; replug |
| Keys ignored | Wayland session, or run over SSH | Log in with **Ubuntu on Xorg** (§1.8) and run from a local terminal |
| Cameras not listed, or permission errors from RealSense | udev rules missing | §11.1, then replug |
| `Couldn't resolve requests` / no frames, with cameras attached | Camera on USB 2, or wrong serial | Use a USB 3 port (§11.2); check `serials.py` (§11.4) |
| `Couldn't get frame for device` during teleop | Camera dropped out: USB 2, a loose cable, or too many devices on one USB controller | USB 3 port, secure the cable, put the cameras on different ports |
| Wrist and front swapped in the video | Serials swapped | Swap them in `serials.py` |

---

## 14. What this test does not cover

- **π_real rollout** (`minimal.py`) uses the same stack. If teleop passes, rollout needs:
  - The same IP and constructor edits.
  - A GPU env with this Polymetis client, or the policy running on another machine.
  - The fixes listed in REAL_ROBOT_FR3_GUIDE.md §7, including the rollout timing caveat.
- **Tracking quality vs. sim.** The original `DiffIKWrapper` runs diff-IK **once per 10 Hz command**, while the sim recomputes it at every physics substep. Teleop working doesn't mean the arm tracks like sim. If that matters, check it with the sim-replay test in REAL_ROBOT_FR3_GUIDE.md §8 (Step 8e), which works the same through Polymetis.
- **Long sessions without an RT kernel.** Path B is checked only for "does it run". Whether its stop rate is acceptable for collecting many demos is something you find out in Phases 9 and 11.
- **Loop timing with cameras.** Without cameras, each 10 Hz cycle has more spare time. Phase 11 checks that the cameras keep up, but not how much they delay each cycle.
