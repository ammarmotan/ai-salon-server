import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
              400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
LIPS_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
LEFT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LEFT_BROW   = [70,63,105,66,107,55,65,52,53,46]
RIGHT_BROW  = [300,293,334,296,336,285,295,282,283,276]
LEFT_CHEEK  = [116,123,147,213,192,214,210,211,212,202,204,194]
RIGHT_CHEEK = [345,352,376,433,411,434,430,431,432,422,424,414]
NOSE_BRIDGE = [6,197,195,5,4,1]
LIPS_INNER  = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

def hex_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))

def get_pts(lm, indices, w, h):
    return np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in indices], np.int32)

def detect_landmarks(img):
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
    )
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with mp_vision.FaceLandmarker.create_from_options(opts) as det:
        res = det.detect(mp_img)
    return res.face_landmarks[0] if res.face_landmarks else None

def soft_mask(shape, pts, blur=15):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [pts], 255)
    b = blur | 1
    return cv2.GaussianBlur(m, (b, b), 0).astype(np.float32) / 255.0

def blend(img, color, mask_f):
    c = np.array(color, np.float32)
    m = mask_f[:, :, None]
    return np.clip(img.astype(np.float32) * (1 - m) + c * m, 0, 255).astype(np.uint8)

def apply_hair_color(img, lm, w, h, hair_color, intensity):
    oval  = get_pts(lm, FACE_OVAL, w, h)
    fx, fy, fw, fh = cv2.boundingRect(oval)

    face_m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(face_m, [oval], 255)

    hair_m = np.zeros((h, w), np.uint8)
    top_y  = max(0, fy - int(fh * 0.05))
    hair_m[:top_y, max(0, fx - fw//4):min(w, fx + fw + fw//4)] = 255
    side_h = int(fh * 0.5)
    hair_m[fy:fy+side_h, max(0, fx - fw//3):fx + fw//8] = 255
    hair_m[fy:fy+side_h, fx + 7*fw//8:min(w, fx + fw + fw//3)] = 255
    hair_m[face_m == 255] = 0
    hair_m = cv2.GaussianBlur(hair_m, (15, 15), 0)

    target = hex_to_bgr(hair_color)
    lab    = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab  = cv2.cvtColor(np.uint8([[list(target)]]), cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
    m      = hair_m.astype(np.float32) / 255.0 * intensity
    lab[:,:,1] = lab[:,:,1] * (1-m) + t_lab[1] * m
    lab[:,:,2] = lab[:,:,2] * (1-m) + t_lab[2] * m
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def apply_lipstick(img, lm, w, h, lip_color, intensity):
    outer      = get_pts(lm, LIPS_OUTER, w, h)
    inner      = get_pts(lm, LIPS_INNER, w, h)
    target_bgr = hex_to_bgr(lip_color)
    # Step 1: solid color blend on full lip area (strong enough to show)
    m_outer = soft_mask(img.shape, outer, blur=3)
    img     = blend(img, target_bgr, m_outer * min(intensity * 1.1, 0.92))
    # Step 2: darker inner shadow for depth
    dark = tuple(max(0, int(c*0.55)) for c in target_bgr)
    m_inner = soft_mask(img.shape, inner, blur=3)
    img  = blend(img, dark, m_inner * intensity * 0.30)
    # Step 3: subtle highlight on upper lip center
    rect = cv2.boundingRect(outer)
    cx, cy = rect[0]+rect[2]//2, rect[1]+rect[3]//4
    hl_pts = np.array([(cx-10,cy),(cx+10,cy),(cx+6,cy+5),(cx-6,cy+5)], np.int32)
    hl_col = tuple(min(255, int(c*1.5)) for c in target_bgr)
    m_hl   = soft_mask(img.shape, hl_pts, blur=9)
    return blend(img, hl_col, m_hl * intensity * 0.22)

def apply_eyebrows(img, lm, w, h, intensity):
    for brow in [LEFT_BROW, RIGHT_BROW]:
        bp   = get_pts(lm, brow, w, h)
        mask = np.zeros((h,w), np.uint8)
        cv2.fillPoly(mask, [bp], 255)
        mean_col = cv2.mean(img, mask=mask)[:3]
        dark = tuple(max(0, int(c*0.30)) for c in mean_col)
        m    = soft_mask(img.shape, bp, blur=5)
        img  = blend(img, dark, m * intensity * 0.72)
    return img

def apply_eyeshadow(img, lm, w, h, intensity, eyeshadow_color="#6B4F62"):
    shadow_bgr = hex_to_bgr(eyeshadow_color)
    shadow_lab = cv2.cvtColor(np.uint8([[list(shadow_bgr)]]), cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
    for eye in [LEFT_EYE, RIGHT_EYE]:
        ep    = get_pts(lm, eye, w, h)
        eye_h = int((ep[:,1].max() - ep[:,1].min()) * 0.9)
        shad  = ep.copy(); shad[:,1] -= eye_h
        lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        m     = soft_mask(img.shape, shad, blur=13) * intensity * 0.40
        lab[:,:,1] = lab[:,:,1]*(1-m) + shadow_lab[1]*m
        lab[:,:,2] = lab[:,:,2]*(1-m) + shadow_lab[2]*m
        lab[:,:,0] = lab[:,:,0]*(1-m*0.2) + shadow_lab[0]*m*0.2
        img = cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return img

def apply_eyeliner(img, lm, w, h, intensity):
    for eye in [LEFT_EYE, RIGHT_EYE]:
        ep    = get_pts(lm, eye, w, h)
        upper = ep[:len(ep)//2+1]
        thick = max(1, int(intensity * 2))
        cv2.polylines(img, [upper], False, (8,5,5), thick, cv2.LINE_AA)
    return img

def apply_blush(img, lm, w, h, intensity, blush_color="#D4847A"):
    blush_bgr = hex_to_bgr(blush_color)
    blush_lab = cv2.cvtColor(np.uint8([[list(blush_bgr)]]), cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
    for cheek in [LEFT_CHEEK, RIGHT_CHEEK]:
        cp  = get_pts(lm, cheek, w, h)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        m   = soft_mask(img.shape, cp, blur=35) * intensity * 0.30
        lab[:,:,1] = lab[:,:,1]*(1-m) + blush_lab[1]*m
        lab[:,:,2] = lab[:,:,2]*(1-m) + blush_lab[2]*m
        img = cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return img

def apply_highlighter(img, lm, w, h, intensity):
    # Brighten L channel only on nose bridge and brow bone
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    nb  = get_pts(lm, NOSE_BRIDGE, w, h)
    m   = soft_mask(img.shape, nb, blur=13) * intensity * 0.20
    lab[:,:,0] = np.clip(lab[:,:,0] + m*18, 0, 255)
    for brow in [LEFT_BROW, RIGHT_BROW]:
        bp = get_pts(lm, brow, w, h).copy(); bp[:,1] -= 5
        m2 = soft_mask(img.shape, bp, blur=11) * intensity * 0.15
        lab[:,:,0] = np.clip(lab[:,:,0] + m2*14, 0, 255)
    return cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def apply_skin_smooth(img, lm, w, h, strength):
    if strength < 0.05:
        return img
    d = max(5, int(strength * 18))
    d = d if d % 2 == 1 else d + 1
    blurred = cv2.bilateralFilter(img, d, 55, 55)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [get_pts(lm, FACE_OVAL, w, h)], 255)
    for idx in [LEFT_EYE, RIGHT_EYE, LIPS_OUTER, LEFT_BROW, RIGHT_BROW]:
        cv2.fillPoly(mask, [get_pts(lm, idx, w, h)], 0)
    m = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0 * strength
    return np.clip(img.astype(np.float32) * (1 - m[:,:,None]) + blurred.astype(np.float32) * m[:,:,None], 0, 255).astype(np.uint8)

def apply_face_filter(img, filter_name, intensity):
    lut = img.copy().astype(np.int16)
    if   filter_name == "warm":    lut[:,:,2]+=20; lut[:,:,1]+=8;  lut[:,:,0]-=10
    elif filter_name == "cool":    lut[:,:,0]+=20; lut[:,:,2]-=10
    elif filter_name == "bright":  lut = (lut * 1.12).astype(np.int16)
    elif filter_name == "vintage": lut[:,:,2]+=15; lut[:,:,1]-=5;  lut[:,:,0]-=15
    elif filter_name == "soft":    lut = (lut * 0.95 + 12).astype(np.int16)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.addWeighted(img, 1-intensity*0.5, lut, intensity*0.5, 0)

def apply_bridal_filter(img, style, intensity):
    """Color grade overlay for bridal looks."""
    lut = img.copy().astype(np.int16)
    if   style == "bridal_royal":    lut[:,:,2]+=20; lut[:,:,1]+=8
    elif style == "bridal_soft":     lut[:,:,2]+=15; lut[:,:,0]-=8
    elif style == "bridal_golden":   lut[:,:,2]+=28; lut[:,:,1]+=14; lut[:,:,0]-=12
    elif style == "bridal_ethereal": lut[:,:,0]+=18; lut[:,:,2]-=8
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.addWeighted(img, 1-intensity*0.4, lut, intensity*0.4, 0)

def process_image(input_path, output_path,
                  hair_color="#503200",
                  lip_color="#c73b3b", eyeshadow_color="#6B4F62",
                  blush_color="#D4847A", smoothness=0.5, intensity=0.6,
                  bridal_style="none", face_filter="none",
                  do_hair=False, do_makeup=False, do_bridal=False, do_filter=False):

    img = cv2.imread(input_path)
    if img is None:
        return

    h0, w0 = img.shape[:2]
    if max(h0, w0) > 1200:
        s = 1200 / max(h0, w0)
        img = cv2.resize(img, (int(w0*s), int(h0*s)))

    h, w = img.shape[:2]
    lm   = detect_landmarks(img)

    if lm is None:
        cv2.imwrite(output_path, img)
        return

    if do_hair:
        img = apply_hair_color(img, lm, w, h, hair_color, intensity)

    if do_makeup:
        img = apply_skin_smooth(img, lm, w, h, smoothness)
        img = apply_eyebrows(img, lm, w, h, intensity)
        img = apply_eyeshadow(img, lm, w, h, intensity, eyeshadow_color)
        img = apply_eyeliner(img, lm, w, h, intensity)
        img = apply_blush(img, lm, w, h, intensity, blush_color)
        img = apply_highlighter(img, lm, w, h, intensity)
        img = apply_lipstick(img, lm, w, h, lip_color, intensity)

    if do_bridal and bridal_style not in ("none", ""):
        img = apply_bridal_filter(img, bridal_style, intensity)

    if do_filter and face_filter not in ("none", ""):
        img = apply_face_filter(img, face_filter, intensity)

    cv2.imwrite(output_path, img)
