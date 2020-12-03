import Slider from '@material-ui/core/Slider';
import React from 'react';
import { updateParameter } from '../actions/parametersActions';
import { updateInterpolatedImage } from '../actions/imagesActions';
import { connect } from 'react-redux';
import axios from 'axios';
import { SERVER_URL } from '../constants';
import _ from 'lodash';

const { Component } = require('react');

class ParameterComponent extends Component {
    constructor(props) {
        super(props);

        this.handleChangeParameter = this.handleChangeParameter.bind(this);
        this.requestInterpolatedImage = _.throttle(this.requestInterpolatedImage, 200);
    }

    handleChangeParameter(newValue) {
        this.props.updateParameter(this.props.index, newValue);

        const parameters = [...this.props.parameters];
        parameters[this.props.index] = newValue;

        this.requestInterpolatedImage(parameters)
    }

    requestInterpolatedImage(parameters) {
        const self = this;
        axios.post(`${SERVER_URL}/interpolate`, { parameters })
            .then(function (response) {
                self.props.updateInterpolatedImage(response.data.image);
            })
            .catch(function (error) {
                // handle error
                console.log(error);
            });
    }

    render() {
        return <div>
            <span>{this.props.name}</span>
            <Slider value={this.props.value}
                    min={-5} max={5} step={0.1}
                    onChange={(event, newValue) =>
                        this.handleChangeParameter(newValue)}
                    orientation="horizontal"
                    aria-labelledby="continuous-slider"/>
        </div>
    }
}

const mapStateToProps = state => ({
    parameters: state.parameters.parameters
});

const mapDispatchToProps = dispatch => ({
    updateParameter: (index, newValue) => dispatch(updateParameter(index, newValue)),
    updateInterpolatedImage: (imageData) => dispatch(updateInterpolatedImage(imageData))
});

export default connect(mapStateToProps, mapDispatchToProps)(ParameterComponent);