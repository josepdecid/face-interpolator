import { UPDATE_INTERPOLATED_IMG, UPDATE_ORIGINAL_IMG } from '../actions/action_types';

export default function imagesReducer(state = { originalImage: null, interpolatedImage: null }, action) {
    switch (action.type) {
        case UPDATE_ORIGINAL_IMG:
            return {
                originalImage: action.imageData,
                interpolatedImage: state.interpolatedImage
            };
        case UPDATE_INTERPOLATED_IMG:
            return {
                originalImage: state.originalImage,
                interpolatedImage: action.imageData
            };
        default:
            return state;
    }
}