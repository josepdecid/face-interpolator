import React, { Component } from 'react';
import axios from 'axios';
import { SERVER_URL } from '../constants';
import { setDefaultParameters } from '../actions/parametersActions';
import { connect } from 'react-redux';
import { swapCameraStatus, updateInterpolatedImage, updateOriginalImage } from '../actions/imagesActions';
import Fab from '@material-ui/core/Fab';
import Camera from 'react-html5-camera-photo';
import PublishIcon from '@material-ui/icons/Publish';
import PhotoCameraIcon from '@material-ui/icons/PhotoCamera';
import Dialog from '@material-ui/core/Dialog';
import DialogContent from '@material-ui/core/DialogContent';
import DialogActions from '@material-ui/core/DialogActions';
import Button from '@material-ui/core/Button';


class ActionButtons extends Component {
    constructor(props) {
        super(props);

        this.handleSwapCameraStatus = this.handleSwapCameraStatus.bind(this);
        this.handleSubmitImage = this.handleSubmitImage.bind(this);
    }

    handleTakePhotoss(dataUri) {
        // Do stuff with the photo...
        console.log(dataUri);
    }

    handleSwapCameraStatus(event) {
        event.preventDefault();
        this.props.swapCameraStatus();
    }

    handleSubmitImage(event) {
        event.preventDefault();

        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = () => {
            this.props.updateOriginalImage(reader.result);
        }
        reader.readAsDataURL(file);

        const data = new FormData();
        data.append('imageData', file);
        const self = this;
        axios.post(`${SERVER_URL}/parametrize`, data)
            .then(function (response) {
                const { parameters, attributeNames, maxVarianceIdx } = response.data;
                self.props.setDefaultParameters(parameters, attributeNames, maxVarianceIdx);
                self.updateInterpolatedImage(self, parameters);
            })
            .catch(function (error) {
                // handle error
                console.log(error);
            })
    }

    updateInterpolatedImage(self, parameters) {
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
        const styles = {
            margin: 0,
            top: 'auto',
            right: 20,
            left: 'auto',
            position: 'fixed',
        };

        return <div>
            <Fab color="primary" component="span" aria-label="Upload image"
                 onClick={this.handleSwapCameraStatus} style={{ ...styles, bottom: 80 }}>
                <PhotoCameraIcon/>
            </Fab>

            <label htmlFor="upload-photo" style={{ ...styles, bottom: 20 }}>
                <input
                    onChange={this.handleSubmitImage}
                    style={{ display: 'none' }}
                    id="upload-photo"
                    name="upload-photo"
                    type="file"/>

                <Fab color="primary" component="span" aria-label="Upload image">
                    <PublishIcon/>
                </Fab>
            </label>

            <Dialog aria-labelledby="customized-dialog-title" open={this.props.isCameraOpen} fullWidth={true}>
                <DialogContent>
                    <Camera/>
                </DialogContent>
                <DialogActions>
                    <Button autoFocus color="secondary" onClick={this.handleSwapCameraStatus}>
                        Cancel
                    </Button>
                    <Button autoFocus color="primary">
                        Take picture
                    </Button>
                </DialogActions>
            </Dialog>
        </div>;
    }
}

const mapStateToProps = state => ({
    isCameraOpen: state.images.isCameraOpen
});

const mapDispatchToProps = dispatch => ({
    setDefaultParameters: (parameters, attributeNames, maxVarianceIdx) =>
        dispatch(setDefaultParameters(parameters, attributeNames, maxVarianceIdx)),
    updateOriginalImage: (imageData) => dispatch(updateOriginalImage(imageData)),
    updateInterpolatedImage: (imageData) => dispatch(updateInterpolatedImage(imageData)),
    swapCameraStatus: () => dispatch(swapCameraStatus())
});

export default connect(mapStateToProps, mapDispatchToProps)(ActionButtons);