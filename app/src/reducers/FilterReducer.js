import { UPDATE_FILTER_PARAMS, UPDATE_FILTER_TEXT } from '../actions/action_types';

const defaultState = {
    text: '',
    values: {
        showAttributes: true,
        showParameters: true,
        sortByVariance: true
    }
};


export default function filterReducer(state = defaultState, action) {
    switch (action.type) {
        case UPDATE_FILTER_TEXT:
            return { ...state, text: action.text };
        case UPDATE_FILTER_PARAMS:
            const { showAttributes, showParameters, sortByVariance } = action.parameters;
            return { ...state, values: { showAttributes, showParameters, sortByVariance } }
        default:
            return state;
    }
};