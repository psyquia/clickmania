import React from 'react';
import { Link } from 'react-router-dom';
import { Navbar } from 'components';
import { css } from '@emotion/css';
import ClickerApp from './ClickerApp';

const currencyName = 'Coin';

const Clicker = () => (
	<>
		<Navbar />
		<div
			className={css`
				width: 100%;
				display: flex;
				flex-direction: column;
				justify-content: center;
				align-items: center;
				text-align: center;
				margin-top: 30px;
			`}
		>
			<h1>{currencyName} Clicker</h1>
			<ClickerApp currencyName={currencyName} />
			<Link to="/">Back to homepage</Link>
		</div>
	</>
);

export default Clicker;
