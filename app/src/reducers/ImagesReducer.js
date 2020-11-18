import { SWAP_CAMERA_STATUS, UPDATE_INTERPOLATED_IMG, UPDATE_ORIGINAL_IMG } from '../actions/action_types';

const defaultState = { originalImage: null, interpolatedImage: null, isCameraOpen: false };


export default function imagesReducer(state = defaultState, action) {
    switch (action.type) {
        case UPDATE_ORIGINAL_IMG:
            return { ...state, originalImage: action.imageData };
        case UPDATE_INTERPOLATED_IMG:
            return { ...state, interpolatedImage: action.imageData };
        case SWAP_CAMERA_STATUS:
            return { ...state, isCameraOpen: !state.isCameraOpen };
        default:
            return state;
    }
};