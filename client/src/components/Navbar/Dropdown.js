import React, { useState, useEffect } from 'react';
import { FaUserCircle } from 'react-icons/fa';
import { css } from '@emotion/css';
import { useUser } from 'context/userContext';
import { useNavigate } from 'react-router-dom';
import { ButtonLink } from './styles';

const Dropdown = () => {
	const { user, setUser } = useUser();
	const navigate = useNavigate();
	const [width, setWidth] = useState(window.innerWidth);
	const breakpoint = 700;
	useEffect(() => {
		const handleResizeWindow = () => setWidth(window.innerWidth);
		// subscribe to window resize event "onComponentDidMount"
		window.addEventListener('resize', handleResizeWindow);
		return () => {
			// unsubscribe "onComponentDestroy"
			window.removeEventListener('resize', handleResizeWindow);
		};
	}, []);

	return width < breakpoint ? (
		<>
			<ButtonLink
				onClick={() => {
					localStorage.setItem('access-token', null);
					setUser(null);
					navigate('/');
				}}
			>
				Log Out
			</ButtonLink>
		</>
	) : (
		<div
			className={css`
				position: relative;
				display: inline-block;
				&:hover {
					display: block;
					> div {
						display: block;
					}
				}
			`}
		>
			<div
				className={css`
					height: 100%;
					width: 100%;
					padding: 1.6rem 2rem;
					display: flex;
					align-items: center;
					justify-content: center;
					margin-top: 4px;
				`}
			>
				<FaUserCircle size={35} />
			</div>
			<div
				className={css`
					display: none;
					position: absolute;
					background-color: #f9f9f9;
					min-width: 230px;
					box-shadow: 0px 8px 16px 0px rgba(0, 0, 0, 0.2);
					z-index: 1;
					right: 0;
					left: auto;
				`}
			>
				<div
					className={css`
						border-color: black;
						border-bottom-width: thin;
						font-size: 2rem;
					`}
				>
					<p className={css``}>{user?.username}</p>
					<p>Coins: {user?.coins}</p>
				</div>

				<ButtonLink
					onClick={() => {
						localStorage.setItem('access-token', null);
						setUser(null);
						navigate('/');
					}}
				>
					Log Out
				</ButtonLink>
			</div>
		</div>
	);
};

export default Dropdown;
