import { RESET_NOTIFICATIONS, SET_NOTIFICATIONS } from '../actions/action_types';

const defaultState = { message: null, mode: null };

export default function notificationsReducer(state = defaultState, action) {
    switch (action.type) {
        case SET_NOTIFICATIONS:
            return { message: action.message, mode: action.mode };
        case RESET_NOTIFICATIONS:
            return defaultState;
        default:
            return state;
    }
};