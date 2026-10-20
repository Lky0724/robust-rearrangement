import furniture_bench  # noqa: F401
import argparse

from src.data_collection.data_collector import DataCollector
from src.common.files import trajectory_save_dir
from src.gym import turn_off_april_tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomness", "-r", type=str, default="low")
    parser.add_argument("--num-demos", "-n", type=int, default=100)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    # parser.add_argument("--resize-sim-img", action="store_true")
    parser.add_argument("--furniture", "-f", type=str, required=True)
    parser.add_argument("--save-failure", action="store_true")
    parser.add_argument("--headless", action="store_true")
    # ResiP's policies, teleop data and RL env all use diffik
    parser.add_argument(
        "--ctrl-mode", "-c", type=str, default="diffik", choices=["diffik", "osc"]
    )
    parser.add_argument(
        "--keep-april-tags",
        action="store_true",
        help="Keep AprilTags on the part meshes (ResiP renders and trains without them)",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Also save an mp4 (wrist | front camera) next to each pickle",
    )

    args = parser.parse_args()

    # Load the tagless part meshes, like the RL env and the rest of the pipeline
    if not args.keep_april_tags:
        turn_off_april_tags()

    # TODO: Consider what we do with images of full size and if that's needed
    # For now, we assume that images are stored in 224x224 and we know that as`image`
    # # Add the suffix _highres if we are not resizing images in or after simulation
    # if not args.resize_img_after_sim and not args.small_sim_img_size:
    #     obs_type = obs_type + "_highres"
    resize_sim_img = False

    data_path = trajectory_save_dir(
        controller=args.ctrl_mode,
        domain="sim",
        task=args.furniture,
        demo_source="scripted",
        randomness=args.randomness,
        suffix=args.suffix,
    )

    print(f"Saving data to directory: {data_path}")

    collector = DataCollector(
        is_sim=True,
        data_path=data_path,
        furniture=args.furniture,
        device_interface=None,
        headless=args.headless,
        manual_label=False,
        scripted=True,
        draw_marker=False,
        randomness=args.randomness,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        resize_sim_img=resize_sim_img,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        ctrl_mode=args.ctrl_mode,
        compress_pickles=True,
        save_mp4=args.save_mp4,
    )

    collector.collect()
