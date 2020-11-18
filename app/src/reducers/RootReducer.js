import { combineReducers } from 'redux';
import imagesReducer from './ImagesReducer';
import parametersReducer from './ParametersReducer';
import notificationsReducer from './NotificationsReducer';
import filterReducer from './FilterReducer';

export default combineReducers({
    images: imagesReducer,
    parameters: parametersReducer,
    notifications: notificationsReducer,
    filter: filterReducer
});