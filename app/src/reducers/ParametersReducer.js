import { SET_PARAMETERS, UPDATE_PARAMETER } from '../actions/action_types';

export default function parametersReducer(state = [], action) {
    switch (action.type) {
        case SET_PARAMETERS:
            return action.parameters;

        case UPDATE_PARAMETER:
            const newParameters = [...state];
            newParameters[action.index] = action.newValue;
            return newParameters;

        default:
            return state;
    }
};