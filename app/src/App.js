import React, { Component } from 'react';
import Grid from '@material-ui/core/Grid';

import './App.css';
import ParametersContainer from './components/ParametersContainer';
import PhotosContainer from './components/PhotosContainer';
import ActionButtons from './components/ActionButtons';

class App extends Component {
    render() {
        return (
            <div className="App">
                <Grid container direction="column" className="App" spacing={2}>
                    <Grid item xs={12} className="photos-container">
                        <PhotosContainer/>
                    </Grid>
                    <Grid item xs={12} className="parameters-container">
                        <ParametersContainer numParameters={5}/>
                    </Grid>
                </Grid>
                <ActionButtons/>
            </div>
        );
    }
}

export default App;
