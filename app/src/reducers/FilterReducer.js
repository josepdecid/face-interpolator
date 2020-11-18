import { UPDATE_FILTER_TEXT } from '../actions/action_types';

const defaultState = { text: '' };


export default function filterReducer(state = defaultState, action) {
    switch (action.type) {
        case UPDATE_FILTER_TEXT:
            return { text: action.text };
        default:
            return state;
    }
};