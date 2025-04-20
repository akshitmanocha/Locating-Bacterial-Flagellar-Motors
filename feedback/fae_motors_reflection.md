@~akshit , you used variety of augmentation, but there are some issues. That is what I am applying on your traing data
1. Only Positive Images Used. The current training data only contains images where a motor exists. The model never sees empty slices, so it learns to always predict a motor, even when none exists.
This causes false positives (wrong detections). In order to fix it, we need to include some negative (empty) slices in our training.
2.  Translation:  I think we need to increase the translate to 0.2.
3.  Perspective Augmentation : The perspective does not make sense for Tomogram slices that are flat (like X-rays). It is better that we set perspective to 0.0.
4. Mosaic: Mosaic helps mix 4 images together for variety. Increasing extra 5 epochs could improve generalization. It is better to use close_mosaic of 15. See the following:

 def train_yolo_model(yaml_path, pretrained_weights_path, epochs=100, 
                     batch_size=25, img_size=960, patience=20):
    """
    Train a YOLO model on the prepared dataset.
    
    Args:
        yaml_path (str): Path to the dataset YAML file.
        pretrained_weights_path (str): Path to pre-downloaded weights file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        img_size (int): Image size for training.
        patience (int): Early stopping patience.
    """
    print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    model = YOLO(pretrained_weights_path)
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=yolo_weights_dir,
        name='motor_detector',
        exist_ok=True,
        patience=patience,
        save_period=5,
        val=True,
        verbose=True,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.00001,
        cos_lr=True,
        weight_decay=0.0005,
        momentum=0.937,
        
        # Core augmentations (spatial)
        degrees=45.0,       
        translate=0.2,      # 1) Increased from 0.1 to 0.2 for larger object shifts (edge visibility)
        scale=0.7,          # 2) Increase from 0.5 to 0.7
        shear=5.0,          # 3) Make it smaller to 5.0 instead of 15 for minor angle variation
        flipud=0.5,         
        fliplr=0.5,         
        
        # RandAugment
        auto_augment="randaugment=m9-n3",  
        
        # Color augmentations 
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        
        # Perspective augmentation
        perspective=0.0,    # 4) Removed perspective warp 
        
        # Advanced augmentations
        mosaic=0.8,         
        close_mosaic=15,    # 5) Extend mosaic augmentation until epoch 15 (was 10) for more mixed-context training
        mixup=0.3,          
        copy_paste=0.3,     
        dropout=0.2,       
        
        # Training efficiency
        amp=True            
    )
    
    run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
    best_epoch_info = plot_dfl_loss_curve(run_dir)
    if best_epoch_info:
        best_epoch, best_val_loss, best_map_50 = best_epoch_info
        print(f"\nBest model found at epoch {best_epoch} with validation "
              f"DFL loss: {best_val_loss:.4f}, and mAP50: {best_map_50:.4f}")
    
    return model
