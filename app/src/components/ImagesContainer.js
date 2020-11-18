import React, { Component } from 'react';
import Grid from '@material-ui/core/Grid';
import { connect } from 'react-redux';
import Paper from '@material-ui/core/Paper';


class ImagesContainer extends Component {
    render() {
        return <Grid container direction="row" justify="center">
            <Grid item xs={5}>
                {this.props.images.originalImage !== null &&
                <Paper elevation={3}>
                    <img width="178" height="218"
                         src={this.props.images.originalImage} alt="Original"/>
                </Paper>}
            </Grid>
            <Grid item xs={5}>
                {this.props.images.interpolatedImage !== null &&
                <Paper elevation={3}>
                    <img width="178" height="218"
                         src={'data:image/jpeg;base64,' + this.props.images.interpolatedImage}
                         key={this.props.images.interpolatedImage} alt="Interpolated"/>
                </Paper>}
            </Grid>
        </Grid>
    }
}

const mapStateToProps = state => ({
    images: state.images,
    parameters: state.parameters
});

export default connect(mapStateToProps)(ImagesContainer);