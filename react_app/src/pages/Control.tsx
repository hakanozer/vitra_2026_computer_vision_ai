import React, { JSX, useEffect, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { apiConfig } from '../services/apiConfig'
import { userProfile } from '../services/userService'
import Navbar from '../components/Navbar'

function Control(props: {item: JSX.Element}) {

  const navigate = useNavigate()
  const jwt = localStorage.getItem('token')
  apiConfig.defaults.headers.common['Authorization'] = `Bearer ${jwt}`
  const [name, setName] = useState('')

  useEffect(() => {
    if (jwt) {
      userProfile()
      .then(res => {
        const dt = res.data
        setName(dt.data.name)
      })
      .catch(err => {
        localStorage.removeItem('token')
        navigate('/', {replace: true})
      })
    }
  }, [])

  return (
    jwt 
        ?
        <>
            <Navbar name={name} />
            {props.item}
        </>
        :
        <Navigate to={'/'} replace={true} />
  )
}

export default Control