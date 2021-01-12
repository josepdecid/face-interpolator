import { SET_PARAMETERS, UPDATE_PARAMETER } from '../actions/action_types';

const defaultState = {
    parameters: [],
    attributeNames: [],
    maxVarianceIdx: []
}

export default function parametersReducer(state = defaultState, action) {
    switch (action.type) {
        case SET_PARAMETERS:
            return {
                parameters: action.parameters,
                attributeNames: action.attributeNames,
                maxVarianceIdx: action.maxVarianceIdx
            };

        case UPDATE_PARAMETER:
            const newParameters = [...state.parameters];
            newParameters[action.index] = action.newValue;
            return {
                ...state,
                parameters: newParameters,
            }

        default:
            return state;
    }
};