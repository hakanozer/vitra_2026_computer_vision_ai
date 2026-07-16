import React, { FormEvent, useRef, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { isValidEmail } from '../utils/valids'
import { ToastContainer, toast } from 'react-toastify';
import { userLogin } from '../services/userService';
import { apiConfig } from '../services/apiConfig';

function Login() {

  // refs
  const emailRef = useRef<HTMLInputElement>(null)
  const passwordRef = useRef<HTMLInputElement>(null)

  const navigate = useNavigate()

  const [email, setEmail] = useState('hakanozer02@gmail.com')
  const [password, setPassword] = useState('123456')
  const sendLogin = (evt: FormEvent) => {
    evt.preventDefault()
    if (!isValidEmail(email)) {
      toast.error('Email format fail!')
      emailRef.current?.focus()
    }else if(password.length < 5) {
      toast.error('Password fail')
      passwordRef.current?.focus()
    }else {
      // service call
      userLogin(email, password)
      .then(res => {
        // servisten 200 döndü, servis başarılı oldu
        const dt = res.data
        localStorage.setItem('token', dt.data.access_token)
        apiConfig.defaults.headers.common['Authorization'] = `Bearer ${dt.data.access_token}`
        // redirect
        // window.location.href = '/products'
        navigate('/products', {replace: true})
      })
      .catch(err => {
        // servis hatalı ise
        toast.error('Incorrect email or password')
      })
    }

  }

  return (
    <>
      <div className="row">
        <div className='col-sm-4'></div>
        <div className='col-sm-4'>
          <h2>User Login</h2>
          <form onSubmit={sendLogin}>
            <div className='mb-3'>
              <input ref={emailRef} value={email} onChange={(evt) => setEmail(evt.target.value)} type='email' className='form-control' placeholder='E-Mail' />
            </div>
            <div className='mb-3'>
              <input ref={passwordRef} value={password} onChange={(evt) => setPassword(evt.target.value)}  type='password' className='form-control' placeholder='Password' />
            </div>
            <div className='d-flex justify-content-between'>
              <button className='btn btn-success'>Login</button>
              <NavLink to={'/register'} className='btn btn-danger'>Register</NavLink>
            </div>
          </form>
        </div>
        <div className='col-sm-4'></div>
      </div>
      <ToastContainer />
    </>
  )
}

export default Login