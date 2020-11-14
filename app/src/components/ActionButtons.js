import React, { Component } from 'react';
import axios from 'axios';
import { SERVER_URL } from '../constants';
import { setDefaultParameters } from '../actions/parametersActions';
import { connect } from 'react-redux';
import { updateInterpolatedImage, updateOriginalImage } from '../actions/imagesActions';


class ActionButtons extends Component {
    constructor(props) {
        super(props);

        this.handleSubmitImage = this.handleSubmitImage.bind(this);
    }

    handleSubmitImage(event) {
        event.preventDefault();

        const file = this.originalImage.files[0];
        const reader = new FileReader();
        reader.onload = () => {
            this.props.updateOriginalImage(reader.result);
        }
        reader.readAsDataURL(file);

        const data = new FormData();
        data.append('imageData', this.originalImage.files[0]);
        const self = this;
        axios.post(`${SERVER_URL}/parametrize`, data)
            .then(function (response) {
                const parameters = response.data.parameters;
                self.props.setDefaultParameters(parameters);
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
        return <div>
            <form onSubmit={this.handleSubmitImage}>
                <input type="file" ref={(ref) => {
                    this.originalImage = ref;
                }}/>
                <input type="submit" value="Submit"/>
            </form>
        </div>;
    }
}

const mapDispatchToProps = dispatch => ({
    setDefaultParameters: (parameters) => dispatch(setDefaultParameters(parameters)),
    updateOriginalImage: (imageData) => dispatch(updateOriginalImage(imageData)),
    updateInterpolatedImage: (imageData) => dispatch(updateInterpolatedImage(imageData))
});

export default connect(null, mapDispatchToProps)(ActionButtons);

/*
            <Fab aria-label="Take photo" color="primary" className="fab" onClick={handleClickOpen}>
                <CameraEnhanceIcon/>
            </Fab>

<Dialog onClose={handleClose} aria-labelledby="customized-dialog-title" open={open}>
                <DialogContent>
                    <Camera
                        onTakePhoto={(dataUri) => {
                            this.handleTakePhoto(dataUri);
                        }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button autoFocus onClick={handleClose} color="primary">
                        Take picture
                    </Button>
                </DialogActions>
            </Dialog>
 */