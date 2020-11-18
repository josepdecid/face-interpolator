import { SET_NOTIFICATIONS, RESET_NOTIFICATIONS } from './action_types';

export function setNotification(message, mode) {
    return { type: SET_NOTIFICATIONS, message, mode };
}

export function resetNotification() {
    return { type: RESET_NOTIFICATIONS };
}