import React, { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { singleProduct } from '../services/productService'
import { iSingleProduct } from '../models/iAllProduct'
import { likesArrControl, likesControl, likeStoreAddRemove } from '../utils/likesStore'
import { useLikesStore } from '../store/useLikesStore'

function ProductDetail() {

  const { changeLikesArr } = useLikesStore()
  
  const [likeIcon, setLikeIcon] = useState(false)
  const [item, setItem] = useState<iSingleProduct>()
  const [mainImage, setMainImage] = useState<string | undefined>()
  const navigate = useNavigate()
  const params = useParams()
  useEffect(() => {
    const id = params.id
    if (id) {
        // service call
        likesStoreControl(id)
        singleProduct(id).then(res => {
          const dt = res.data
          setItem(dt)
          // set initial main image if images exist
          if (dt?.data?.images && dt.data.images.length > 0) setMainImage(dt.data.images[0])
        }).catch(err => {
          navigate('/products', {replace: true})
        })
    }
  }, [params.id, navigate])

  const likeAddRemove = () => {
    setLikeIcon(!likeIcon)
    const id = params.id
    if (id) {
      likeStoreAddRemove(id)
      likesStoreControl(id)
    }
  }

  const likesStoreControl = (id: string) => {
    const likesStatus = likesControl(id)
    setLikeIcon(likesStatus)
    const arr = likesArrControl()
    if (arr) {
      changeLikesArr(arr)
    }
  }

  

  return (
    <div className="container py-4">
      {!item && (
        <div className="d-flex justify-content-center align-items-center" style={{minHeight: '40vh'}}>
          <div className="spinner-border" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      )}

      {item && (
        <div className="row g-4">
          {/* Left column: product details */}
          <div className="col-12 col-lg-6">
            <div className="card h-100">
              <div className="card-body d-flex flex-column">
                <h2 className="card-title display-6 mb-3">{item.data.title}</h2>
                <h5 className="text-muted">{item.data.brand} • {item.data.category}</h5>

                <div className="my-3">
                  <span className="h4 text-primary">₺{item.data.price.toFixed(2)}</span>
                  {item.data.discountPercentage ? (
                    <small className="text-danger ms-2">{item.data.discountPercentage}% off</small>
                  ) : null}
                </div>

                <p className="flex-grow-1">{item.data.description}</p>

                <i onClick={likeAddRemove} className={ likeIcon ? 'bi bi-suit-heart-fill fs-3 text-danger' : 'bi bi-suit-heart fs-3 text-danger'} role='button'></i>

                <ul className="list-group list-group-flush mb-3">
                  <li className="list-group-item">Stock: {item.data.stock}</li>
                  <li className="list-group-item">Rating: {item.data.rating} / 5</li>
                  <li className="list-group-item">SKU: {item.data.sku}</li>
                </ul>

                <div className="d-flex gap-2">
                  <button className="btn btn-primary" onClick={() => navigate('/products')}>Back to Products</button>
                </div>
              </div>
            </div>
          </div>

          {/* Right column: main image + thumbnails */}
          <div className="col-12 col-lg-6">
            <div className="card h-100">
              <div className="card-body d-flex flex-column">
                <div className="mb-3" style={{flex: '1 1 auto'}}>
                  {mainImage ? (
                    <img src={mainImage} alt={item.data.title} className="img-fluid border" style={{width: '100%', height: 'auto', objectFit: 'contain'}} />
                  ) : (
                    <div className="d-flex align-items-center justify-content-center bg-light border" style={{height: '320px'}}>
                      <span className="text-muted">No image available</span>
                    </div>
                  )}
                </div>

                {item.data.images && item.data.images.length > 0 && (
                  <div className="mt-3">
                    <div className="row g-2">
                      {item.data.images.map((img, idx) => (
                        <div key={idx} className="col-4 col-md-3">
                          <button className={`btn w-100 p-0 border ${mainImage === img ? 'border-primary' : ''}`} onClick={() => setMainImage(img)} style={{background: 'transparent'}}>
                            <img src={img} alt={`${item.data.title} ${idx+1}`} className="img-fluid" style={{height: '80px', objectFit: 'cover', width: '100%'}} />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ProductDetail