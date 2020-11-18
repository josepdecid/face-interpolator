import { UPDATE_FILTER_TEXT } from './action_types';

export function updateFilterText(text) {
    return { type: UPDATE_FILTER_TEXT, text };
}