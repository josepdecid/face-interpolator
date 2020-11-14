import { combineReducers } from 'redux';
import imagesReducer from './ImagesReducer';
import parametersReducer from './ParametersReducer';

export default combineReducers({
    images: imagesReducer,
    parameters: parametersReducer
})