def get_ori_face_bbox(shape, face_bbox, body_bbox, padding=(0.23, 0.23, 0.35, 0.1)):
    """
    Transform face bbox from cropped body coordinates to original frame coordinates,
    and compute an expanded bbox with padding.

    Args:
        shape: Original frame dimensions (height, width)
        face_bbox: Face bbox in cropped body image coordinates (x1, y1, x2, y2)
        body_bbox: Body bbox in frame coordinates (x1, y1, x2, y2), used as offset
        padding: Expansion ratios (left, right, top, bottom) relative to face size

    Returns:
        (frame_face_bbox, frame_expanded_bbox): Both in frame coordinates (x1, y1, x2, y2)
    """
    # Unpack face bbox in cropped body image coordinates
    crop_face_x1, crop_face_y1, crop_face_x2, crop_face_y2 = face_bbox

    # Transform from cropped coordinates to frame coordinates by adding body offset
    frame_face_x1 = body_bbox[0] + crop_face_x1
    frame_face_y1 = body_bbox[1] + crop_face_y1
    frame_face_x2 = body_bbox[0] + crop_face_x2
    frame_face_y2 = body_bbox[1] + crop_face_y2

    frame_face_bbox = (frame_face_x1, frame_face_y1, frame_face_x2, frame_face_y2)

    # Calculate face dimensions for expansion
    face_w = crop_face_x2 - crop_face_x1
    face_h = crop_face_y2 - crop_face_y1

    # Calculate expansion amounts (asymmetric: more on top for hair/forehead)
    expand_left = int(face_w * padding[0])    # 23% of width
    expand_right = int(face_w * padding[1])   # 23% of width
    expand_top = int(face_h * padding[2])     # 35% of height
    expand_bottom = int(face_h * padding[3])  # 10% of height

    # Apply expansion to frame coordinates and clamp to frame boundaries
    frame_expanded_x1 = max(0, frame_face_x1 - expand_left)
    frame_expanded_y1 = max(0, frame_face_y1 - expand_top)
    frame_expanded_x2 = min(shape[1], frame_face_x2 + expand_right)
    frame_expanded_y2 = min(shape[0], frame_face_y2 + expand_bottom)

    frame_expanded_bbox = (frame_expanded_x1, frame_expanded_y1, frame_expanded_x2, frame_expanded_y2)

    return frame_face_bbox, frame_expanded_bbox
