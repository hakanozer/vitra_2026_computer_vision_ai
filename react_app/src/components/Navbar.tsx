import React from 'react'
import { userLogout } from '../services/userService'
import { NavLink, useNavigate } from 'react-router-dom'
import { useLikesStore } from '../store/useLikesStore';

function Navbar(props: {name: string}) {

  const { likesArr } = useLikesStore();
  const navigate = useNavigate()  
  const logout = () => {
    const answer = window.confirm("Are you sure logout!")
    if (answer) {
        userLogout().then(res => {
            localStorage.removeItem('token')
            navigate('/', {replace: true})
        })
    }
  }

  
    
  return (
    <nav className="navbar navbar-expand-lg bg-body-tertiary">
    <div className="container-fluid">
        <a className="navbar-brand" href="#">Navbar</a>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
        <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarSupportedContent">
        <ul className="navbar-nav me-auto mb-2 mb-lg-0">
            <li className="nav-item">
                <NavLink to={'/products'} className="nav-link">Products</NavLink>
            </li>
            <li className="nav-item">
                <NavLink to={'/likes'} className="nav-link">Likes</NavLink>
            </li>
            <li className="nav-item">
                <NavLink to={'/users'} className="nav-link">Users</NavLink>
            </li>
            <li className="nav-item dropdown">
            <a className="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown" aria-expanded="false">
                Profile
            </a>
            <ul className="dropdown-menu">
                <li><a className="dropdown-item" href="#">Action</a></li>
                <li><a className="dropdown-item" href="#">Another action</a></li>
                <li><hr className="dropdown-divider"/></li>
                <li><a onClick={logout} role='button' className="dropdown-item">Logout</a></li>
            </ul>
            </li>
            <li className="nav-item">
            <a className="nav-link disabled" aria-disabled="true">Sn. {props.name} - ({likesArr.length})</a>
            </li>
        </ul>
        <form action={'/search'} className="d-flex" role="search">
            <input name='q' className="form-control me-2" type="search" placeholder="Search" aria-label="Search"/>
            <button className="btn btn-outline-success" type="submit">Search</button>
        </form>
        </div>
    </div>
    </nav>
  )
}

export default Navbar