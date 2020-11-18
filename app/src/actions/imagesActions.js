import { SWAP_CAMERA_STATUS, UPDATE_INTERPOLATED_IMG, UPDATE_ORIGINAL_IMG } from './action_types';

export function updateInterpolatedImage(imageData) {
    return { type: UPDATE_INTERPOLATED_IMG, imageData };
}

export function updateOriginalImage(imageData) {
    return { type: UPDATE_ORIGINAL_IMG, imageData };
}

export function swapCameraStatus() {
    return { type: SWAP_CAMERA_STATUS };
}