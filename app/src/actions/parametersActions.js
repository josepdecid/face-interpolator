import { SET_PARAMETERS, UPDATE_PARAMETER } from './action_types';

export function setDefaultParameters(parameters, attributeNames) {
    return { type: SET_PARAMETERS, parameters, attributeNames };
}

export function updateParameter(index, newValue) {
    return { type: UPDATE_PARAMETER, index, newValue };
}