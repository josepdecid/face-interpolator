import React, { Component } from 'react';
import Snackbar from '@material-ui/core/Snackbar';
import MuiAlert from '@material-ui/lab/Alert';
import { connect } from 'react-redux';
import { resetNotification } from '../actions/notificationActions';

function Alert(props) {
    return <MuiAlert elevation={6} variant="filled" {...props} />;
}

class NotificationsContainer extends Component {
    constructor(props) {
        super(props);
        this.handleClose = this.handleClose.bind(this);
    }
    handleClose() {
        this.props.resetNotifications();
    }

    render() {
        const open = this.props.notifications.message !== null;
        return <div>
            <Snackbar open={open} autoHideDuration={3000} onClose={this.handleClose}>
                <Alert onClose={this.handleClose} severity={this.props.notifications.mode}>
                    {this.props.notifications.message}
                </Alert>
            </Snackbar>
        </div>;
    }
}

const mapStateToProps = state => ({
    notifications: state.notifications
});

const mapDispatchToProps = dispatch => ({
    resetNotifications: () => dispatch(resetNotification()),
});

export default connect(mapStateToProps, mapDispatchToProps)(NotificationsContainer);