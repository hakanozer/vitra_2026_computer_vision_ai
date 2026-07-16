import React from 'react'
import { iProduct } from '../models/iAllProduct'
import { NavLink } from 'react-router-dom'

function ProductItem(props: {item: iProduct}) {
  return (
    <div className="card mb-3">
        <img src={props.item.images[0]} className="card-img-top" alt="..." />
        <div className="card-body">
            <h5 className="card-title" style={{height: 60,}}>{props.item.title}</h5>
            <p className="card-text">{props.item.price}₺</p>
            <div className='d-grid'>
                <NavLink to={'/productDetail/'+props.item.id} className="btn btn-light">Detail</NavLink>
            </div>
        </div>
    </div>
  )
}

export default ProductItem