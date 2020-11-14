import React, { Component } from 'react';
import Slider from '@material-ui/core/Slider';
import { setDefaultParameters, updateParameter } from '../actions/parametersActions';
import { connect } from 'react-redux';
import axios from 'axios';
import _ from 'lodash';
import { SERVER_URL } from '../constants';
import { updateInterpolatedImage } from '../actions/imagesActions';

class ParametersContainer extends Component {
    constructor(props) {
        super(props);

        this.handleChangeParameter = this.handleChangeParameter.bind(this);
        this.requestInterpolatedImage = _.throttle(this.requestInterpolatedImage, 200);
    }

    handleChangeParameter(index, newValue) {
        this.props.updateParameter(index, newValue);

        const parameters = [...this.props.parameters];
        parameters[index] = newValue;

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
        return this.props.parameters.map((value, key) => {
            return <Slider value={value}
                           min={-2000} max={2000}
                           onChange={(event, newValue) =>
                               this.handleChangeParameter(key, newValue)}
                           orientation="vertical"
                           key={key}/>
        });
    }
}

const mapStateToProps = state => ({
    parameters: state.parameters
});

const mapDispatchToProps = dispatch => ({
    setDefaultParameters: (parameters) => dispatch(setDefaultParameters(parameters)),
    updateParameter: (index, newValue) => dispatch(updateParameter(index, newValue)),
    updateInterpolatedImage: (imageData) => dispatch(updateInterpolatedImage(imageData))
});

export default connect(mapStateToProps, mapDispatchToProps)(ParametersContainer);