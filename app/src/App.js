import React, { Component } from 'react';
import Grid from '@material-ui/core/Grid';

import './App.css';
import ParametersContainer from './components/ParametersContainer';
import PhotosContainer from './components/ImagesContainer';
import ActionButtons from './components/ActionButtons';
import Divider from '@material-ui/core/Divider';
import SearchAppBar from './components/NavBarComponent';

class App extends Component {
    render() {
        return <div className="App">
            <SearchAppBar/>
            <Grid container direction="column" spacing={2}>
                <Grid item xs={12} className="photos-container">
                    <PhotosContainer/>
                </Grid>
                <Divider light/>
                <Grid item xs={12} className="parameters-container">
                    <ParametersContainer numParameters={5}/>
                </Grid>
            </Grid>
            <ActionButtons/>
        </div>;
    }
}

export default App;
