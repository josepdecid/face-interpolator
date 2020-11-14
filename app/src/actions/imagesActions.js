import { UPDATE_INTERPOLATED_IMG, UPDATE_ORIGINAL_IMG } from './action_types';

export function updateInterpolatedImage(imageData) {
    return { type: UPDATE_INTERPOLATED_IMG, imageData };
}

export function updateOriginalImage(imageData) {
    return { type: UPDATE_ORIGINAL_IMG, imageData };
}