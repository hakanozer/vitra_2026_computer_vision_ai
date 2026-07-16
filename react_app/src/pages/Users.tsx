import React, { useEffect, useMemo, useState } from 'react'
import { allUser } from '../services/usersService'
import { SingleUser } from '../models/iAllUser'

function Users() {
  const [count, setCount] = useState(0)
  const [other, setOther] = useState(0)

  const call = (num: number) => {
    console.log("⚙️ Ağır işlem çalıştı")
    let total = 0
    for (let i = 0; i < 1_000; i++) { // Simülasyon
      total += i
    }
    return num * 10 + total
  }
  const result = useMemo(() => call(count), [count])
  console.log("🌀 Component render oldu")

  const [users, setUsers] = useState<SingleUser[]>([])
  const [selectUser, setSelectUser] = useState<SingleUser>()
  useEffect(() => {
    allUser().then(res => {
      const dt = res.data
      setUsers(dt.data)
    })
  }, [])
  

  return (
    <>
      <button onClick={() => setCount(count + 1)} className='btn btn-danger btn-sm'>
        Count Artır
      </button>
      <button onClick={() => setOther(other + 1)} className='btn btn-info btn-sm'>
        Other Artır
      </button>
      <p>Sonuç: {result}</p>
      <p>Other: {other}</p>
      <hr />
      <h2>Users</h2>

      <table className="table table-hover">
        <thead>
          <tr>
            <th scope="col">ID</th>
            <th scope="col">Profile</th>
            <th scope="col">Name</th>
            <th scope="col">Email</th>
            <th scope="col">Phone</th>
          </tr>
        </thead>
        <tbody>

          {users.map((item, index) => 
            <tr onClick={() => setSelectUser(item)} data-bs-toggle="modal" data-bs-target="#exampleModal" role='button' key={index}>
              <th scope="row">{item.id}</th>
              <td>
                <img style={{width: 88,}} src={item.profile} className='img-thumbnail rounded-circle' />
              </td>
              <td>{item.name}</td>
              <td>{item.email}</td>
              <td>{item.phone}</td>
            </tr>
          )}
          
        </tbody>
      </table>

      <div
  className="modal fade"
  id="exampleModal"
  tabIndex={-1}
  aria-labelledby="exampleModalLabel"
  aria-hidden="true"
>
  <div className="modal-dialog modal-dialog-centered">
    <div className="modal-content">
      <div className="modal-header">
        <h1 className="modal-title fs-5" id="exampleModalLabel">
          {selectUser?.name} ({selectUser?.username})
        </h1>
        <button
          type="button"
          className="btn-close"
          data-bs-dismiss="modal"
          aria-label="Close"
        ></button>
      </div>

      <div className="modal-body">
        <div className="text-center mb-3">
          <img
            src={selectUser?.profile}
            alt={selectUser?.name}
            className="img-thumbnail rounded-circle"
            width={120}
            height={120}
          />
        </div>

        <ul className="list-group text-start">
          <li className="list-group-item">
            <strong>Email:</strong> {selectUser?.email}
          </li>
          <li className="list-group-item">
            <strong>Phone:</strong> {selectUser?.phone}
          </li>
          <li className="list-group-item">
            <strong>Website:</strong>{" "}
            <a href={`https://${selectUser?.website}`} target="_blank" rel="noreferrer">
              {selectUser?.website}
            </a>
          </li>
          <li className="list-group-item">
            <strong>Role:</strong> {selectUser?.role}
          </li>

          <li className="list-group-item">
            <strong>Address:</strong>
            <br />
            {selectUser?.address?.street}, {selectUser?.address?.suite}
            <br />
            {selectUser?.address?.city} / {selectUser?.address?.zipcode}
            <br />
            <small>
              Lat: {selectUser?.address?.geo?.lat}, Lng: {selectUser?.address?.geo?.lng}
            </small>
          </li>

          <li className="list-group-item">
            <strong>Company:</strong>
            <br />
            {selectUser?.company?.name}
            <br />
            <em>{selectUser?.company?.catchPhrase}</em>
            <br />
            <small>{selectUser?.company?.bs}</small>
          </li>
        </ul>
      </div>

      <div className="modal-footer">
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          data-bs-dismiss="modal"
        >
          Close
        </button>
      </div>
    </div>
  </div>
</div>


    </>
  )
}

export default Users
