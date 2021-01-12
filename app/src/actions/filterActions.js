import { UPDATE_FILTER_PARAMS, UPDATE_FILTER_TEXT } from './action_types';

export function updateFilterText(text) {
    return { type: UPDATE_FILTER_TEXT, text };
}

export function updateFilterValues(parameters) {
    return { type: UPDATE_FILTER_PARAMS, parameters };
}