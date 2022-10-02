    # HR part
    hr_img = cv2.imread(HRpath)
    h, w = hr_img.shape[:2]
    
    h_patchs = h//CFG.HR_patch_size
    w_patchs = w//CFG.HR_patch_size
    
    for i in range(0, h, CFG.HR_patch_size):
        for j in range(0, w, CFG.HR_patch_size):
            patch = hr_img[i:i+CFG.HR_patch_size, j:j+CFG.HR_patch_size, :]
            cv2.imwrite(f'./test/hr/{i+CFG.HR_patch_size}_{j+CFG.HR_patch_size}.png', patch)
