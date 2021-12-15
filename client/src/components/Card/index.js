import React, { useState } from 'react';
import './style.css';
import client from 'api/client';
import { css } from '@emotion/css';
import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.min.css';
import { toast } from 'react-toastify';

const Card = ({ rarity, img_url, title, isOnSale, setForceUpdate, collection_name, own }) => {
	const [show, setShow] = useState(false);
	const [price, setPrice] = useState(100);

	const handleClose = () => {
		setPrice(100);
		setShow(false);
	};
	const handleShow = () => setShow(true);

	const handleSellArtwork = async () => {
		try {
			setPrice(100);
			const data = await client.post('auth/create_sale', {
				ownID: own.ownID,
				price: parseFloat(price),
			});
			if (data.data?.success) {
				toast.success('Success!', { autoClose: 1000 });
			}
		} catch (err) {
			const message = err?.response?.data?.detail || 'Oops, something went wrong. Please try again later.';
			toast.error(message);
		} finally {
			setShow(false);
			setForceUpdate();
		}
	};

	return (
		<div className="card-container">
			<div className="card">
				<img className={`artwork-img rarity-${rarity}`} src={client.getImg(img_url)} alt={title} />
				<div className="card-body">
					<h2>{title}</h2>
					<h4>{collection_name}</h4>
				</div>
				<div className="sell-overlay">
					<button
						className={isOnSale ? 'sell-btn-disable' : 'sell-btn'}
						onClick={handleShow}
						disabled={isOnSale}
					>
						{isOnSale ? 'On Sale' : 'Sell'}
					</button>
					<Modal show={show} onHide={handleClose} animation={false}>
						<Modal.Header closeButton>
							<Modal.Title>Sell {`"${title}"`}</Modal.Title>
						</Modal.Header>
						<Modal.Body>
							<div className="form-group">
								<label
									htmlFor="price"
									className={css`
										font-size: 1.5rem;
									`}
								>
									Price
								</label>
								<input
									type="number"
									className={css`
										font-size: 1.5rem;
										margin-left: 15px;
									`}
									id="price"
									value={price}
									onChange={(e) => setPrice(e.target.value)}
								/>
							</div>
						</Modal.Body>
						<Modal.Footer>
							<Button variant="secondary" onClick={handleClose}>
								<h3
									className={css`
										font-size: 1.7rem;
										height: 16px;
									`}
								>
									Cancel
								</h3>
							</Button>
							<Button variant="primary" onClick={handleSellArtwork}>
								<h3
									className={css`
										font-size: 1.7rem;
										height: 16px;
									`}
								>
									Sell
								</h3>
							</Button>
						</Modal.Footer>
					</Modal>
				</div>
			</div>
		</div>
	);
};

export default Card;
