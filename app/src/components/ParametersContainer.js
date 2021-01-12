import React, { Component } from 'react';
import { setDefaultParameters } from '../actions/parametersActions';
import { connect } from 'react-redux';
import ParameterComponent from './ParameterComponent';
import Divider from '@material-ui/core/Divider';

class ParametersContainer extends Component {
    render() {
        const attributeSliders = this.props.attributeNames
            .map((name, key) => {
                const value = this.props.parameters[key];
                if (name.toLowerCase().includes(this.props.filterValue.toLowerCase())) {
                    return <ParameterComponent name={name} value={value} key={key} index={key}/>;
                } else return <span/>;
            });

        let parameterSliders;
        if (this.props.filterValues.sortByVariance) {
            parameterSliders = this.props.maxVarianceIdx
                .map((index) => {
                    const name = `Parameter #${index}`
                    const value = this.props.parameters[this.props.attributeNames.length + index]
                    if (name.toLowerCase().includes(this.props.filterValue.toLowerCase())) {
                        return <ParameterComponent name={name} value={value} key={index} index={index}/>;
                    } else return <span/>;
                });
        } else {
            parameterSliders = this.props.parameters
                .map((value, key) => {
                    if (key < this.props.attributeNames.length) return <span/>;
                    const name = `Parameter #${key}`
                    if (name.toLowerCase().includes(this.props.filterValue.toLowerCase())) {
                        return <ParameterComponent name={name} value={value} key={key} index={key}/>;
                    } else return <span/>;
                });
        }

        return <div style={{ overflowY: 'scroll', height: '100%' }}>
            {this.props.filterValues.showAttributes && attributeSliders}
            <Divider light/>
            {this.props.filterValues.showParameters && parameterSliders}
        </div>
    }
}

const mapStateToProps = state => ({
    parameters: state.parameters.parameters,
    attributeNames: state.parameters.attributeNames,
    maxVarianceIdx: state.parameters.maxVarianceIdx,
    filterValue: state.filter.text,
    filterValues: state.filter.values
});

const mapDispatchToProps = dispatch => ({
    setDefaultParameters: (parameters) => dispatch(setDefaultParameters(parameters))
});

export default connect(mapStateToProps, mapDispatchToProps)(ParametersContainer);