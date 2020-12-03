import { SET_PARAMETERS, UPDATE_PARAMETER } from '../actions/action_types';

const defaultState = {
    parameters: [],
    attributeNames: []
}

export default function parametersReducer(state = defaultState, action) {
    switch (action.type) {
        case SET_PARAMETERS:
            return {
                parameters: action.parameters,
                attributeNames: action.attributeNames
            };

        case UPDATE_PARAMETER:
            const newParameters = [...state.parameters];
            newParameters[action.index] = action.newValue;
            return {
                parameters: newParameters,
                attributeNames: [...state.attributeNames]
            }

        default:
            return state;
    }
};