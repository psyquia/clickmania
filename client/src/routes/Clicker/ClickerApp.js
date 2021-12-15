import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { css } from '@emotion/css';
import im from 'assets/coin.png';
import { depositCurrency } from 'api/apiCalls';
import { useUser } from '../../context/userContext';
import './ClickerApp.css';

const ClickerApp = ({ currencyName }) => {
	const [score, setScore] = useState(0);
	const [balance, setBalance] = useState(0);
	const { user, setUser } = useUser();

	const handleDepositCurrency = async () => {
		if (!score) return;

		// api call to deposit coins
		const res = await depositCurrency(user, score);

		if (res.id) {
			toast.success('Success!', { autoClose: 1000 });
			setBalance(res.coins);
			setUser(res);
		} else {
			toast.error('Oops, something went wrong. Please try again later.');
			// return early to prevent clearing score
			return;
		}
		// reset score
		setScore(0);
	};

	useEffect(() => {
		setBalance(user?.coins);
	}, []);

	return (
		user && (
			<div className="clicker-app-container">
				<div // sidebar
					className="score-bar"
				>
					<div className="current-score">
						{score}
						<br /> {currencyName}s
					</div>
					<div className="total-score">
						Total {currencyName}s:
						<br /> {balance}
					</div>
				</div>
				<div className="image-container">
					<input
						type="image"
						src={im}
						alt={currencyName}
						className={css`
							width: 60%;
							transition: width 0.065s;
							&:active {
								width: 50%;
							}
						`}
						onClick={() => setScore((e) => e + 1)}
						onKeyPress={(e) => {
							if (e.key === 'Space') setScore((v) => v + 1);
						}}
					/>
				</div>
				<div // sidebar
					className="score-bar bottom-border"
				>
					<button type="button" className="deposit-btn" onClick={handleDepositCurrency}>
						Deposit
					</button>
				</div>
			</div>
		)
	);
};

export default ClickerApp;
