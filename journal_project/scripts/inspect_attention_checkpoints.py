from ultralytics import YOLO


models = {
    "A1": (
        "journal_project/experiments/"
        "adaptive_dual_attention_ablation/"
        "yolo11s_adaptive_dual_attention_p4/"
        "weights/best.pt"
    ),
    "A2": (
        "journal_project/experiments/"
        "adaptive_dual_attention_ablation/"
        "yolo11s_adaptive_dual_attention_p4_positive_gate/"
        "weights/best.pt"
    ),
}


for name, path in models.items():
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    model = YOLO(path)

    for module in model.model.modules():
        if module.__class__.__name__ == "AdaptiveDualAttention":
            print("Attributes:")
            for key, value in module.__dict__.items():
                if key.startswith("_"):
                    continue
                print(f"  {key}: {value}")

            print("\nParameters:")
            for key, value in module.named_parameters():
                print(
                    f"  {key}: "
                    f"shape={tuple(value.shape)} "
                    f"value={value.detach().cpu().flatten()[:10]}"
                )

            print("\nChildren:")
            for key, value in module.named_children():
                print(f"  {key}: {value}")

            break