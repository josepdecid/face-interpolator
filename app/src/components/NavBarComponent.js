import React, { Component } from 'react';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import IconButton from '@material-ui/core/IconButton';
import Typography from '@material-ui/core/Typography';
import InputBase from '@material-ui/core/InputBase';
import { fade, withStyles } from '@material-ui/core/styles';
import MenuIcon from '@material-ui/icons/Menu';
import SearchIcon from '@material-ui/icons/Search';
import { connect } from 'react-redux';
import { updateFilterText, updateFilterValues } from '../actions/filterActions';
import Switch from '@material-ui/core/Switch';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';

const useStyles = theme => ({
    root: {
        flexGrow: 1,
    },
    menuButton: {
        marginRight: theme.spacing(2),
    },
    title: {
        flexGrow: 1,
        display: 'none',
        [theme.breakpoints.up('sm')]: {
            display: 'block',
        },
    },
    search: {
        position: 'relative',
        borderRadius: theme.shape.borderRadius,
        backgroundColor: fade(theme.palette.common.white, 0.15),
        '&:hover': {
            backgroundColor: fade(theme.palette.common.white, 0.25),
        },
        marginLeft: 0,
        width: '100%',
        [theme.breakpoints.up('sm')]: {
            marginLeft: theme.spacing(1),
            width: 'auto',
        },
    },
    searchIcon: {
        padding: theme.spacing(0, 2),
        height: '100%',
        position: 'absolute',
        pointerEvents: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    inputRoot: {
        color: 'inherit',
    },
    inputInput: {
        padding: theme.spacing(1, 1, 1, 0),
        // vertical padding + font size from searchIcon
        paddingLeft: `calc(1em + ${theme.spacing(4)}px)`,
        transition: theme.transitions.create('width'),
        width: '100%',
        [theme.breakpoints.up('sm')]: {
            width: '12ch',
            '&:focus': {
                width: '20ch',
            },
        },
    },
});

class SearchAppBar extends Component {
    constructor(props) {
        super(props);
        this.handleChangeFilterText = this.handleChangeFilterText.bind(this);
    }

    handleChangeFilterText(event) {
        event.preventDefault();
        this.props.updateFilterText(event.target.value);
    }

    handleChangeFilterValues(event, key) {
        event.preventDefault();

        const parameters = this.props.filterValues;
        parameters[key] = event.target.checked;
        this.props.updateFilterValues(parameters);
    }

    render() {
        const { classes } = this.props;
        return (
            <div className={classes.root}>
                <AppBar position="static">
                    <Toolbar>
                        <IconButton
                            edge="start"
                            className={classes.menuButton}
                            color="inherit"
                            aria-label="open drawer"
                        >
                            <MenuIcon/>
                        </IconButton>
                        <Typography className={classes.title} variant="h6" noWrap>
                            Face Interpolator
                        </Typography>
                        <FormControlLabel
                            control={<Checkbox name="Show Attributes"
                                               checked={this.props.filterValues.showAttributes}
                                               onChange={ev => this.handleChangeFilterValues(ev, 'showAttributes')}/>}
                            label="Show Attributes"
                        />

                        <FormControlLabel
                            control={<Checkbox name="Show Parameters"
                                               checked={this.props.filterValues.showParameters}
                                               onChange={ev => this.handleChangeFilterValues(ev, 'showParameters')}/>}
                            label="Show Parameters"
                        />

                        <FormControlLabel
                            control={<Checkbox name="Sort by Variance"
                                               checked={this.props.filterValues.sortByVariance}
                                               onChange={ev => this.handleChangeFilterValues(ev, 'sortByVariance')}/>}
                            label="Sort by Variance"
                        />

                        <div className={classes.search}>
                            <div className={classes.searchIcon}>
                                <SearchIcon/>
                            </div>
                            <InputBase
                                value={this.props.filterText}
                                onChange={this.handleChangeFilterText}
                                placeholder="Filter…"
                                classes={{
                                    root: classes.inputRoot,
                                    input: classes.inputInput,
                                }}
                                inputProps={{ 'aria-label': 'filter' }}
                            />
                        </div>
                    </Toolbar>
                </AppBar>
            </div>
        );
    }
}

const mapStateToProps = state => ({
    filterText: state.filter.text,
    filterValues: state.filter.values
});

const mapDispatchToProps = dispatch => ({
    updateFilterText: (text) => dispatch(updateFilterText(text)),
    updateFilterValues: (parameters) => dispatch(updateFilterValues(parameters))
});

SearchAppBar = withStyles(useStyles)(SearchAppBar);
export default connect(mapStateToProps, mapDispatchToProps)(SearchAppBar);