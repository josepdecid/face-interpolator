import { SET_PARAMETERS, UPDATE_PARAMETER } from './action_types';

export function setDefaultParameters(parameters, attributeNames, maxVarianceIdx) {
    return { type: SET_PARAMETERS, parameters, attributeNames, maxVarianceIdx };
}

export function updateParameter(index, newValue) {
    return { type: UPDATE_PARAMETER, index, newValue };
}