import React, { Component } from 'react';
import { setDefaultParameters } from '../actions/parametersActions';
import { connect } from 'react-redux';
import ParameterComponent from './ParameterComponent';

class ParametersContainer extends Component {
    render() {
        const sliders = this.props.parameters
            .map((value, key) => {
                if (`Parameters_${key}`.toLowerCase().includes(this.props.filterValue.toLowerCase())) {
                    return <ParameterComponent name={`Parameter_${key}`} value={value} key={key} index={key}/>;
                } else {
                    return <span/>;
                }
            });

        return <div style={{ overflowY: 'scroll', height: '100%' }}>
            {sliders}
        </div>
    }
}

const mapStateToProps = state => ({
    parameters: state.parameters,
    filterValue: state.filter.text
});

const mapDispatchToProps = dispatch => ({
    setDefaultParameters: (parameters) => dispatch(setDefaultParameters(parameters))
});

export default connect(mapStateToProps, mapDispatchToProps)(ParametersContainer);