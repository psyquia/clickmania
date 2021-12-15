import React, { useState, useEffect } from 'react';
import { css } from '@emotion/css';
import { Link, useNavigate } from 'react-router-dom';
import layer1RightIm2 from 'assets/homeRtwo.png';
import client from 'api/client';
import { useUser } from '../../context/userContext';
import './layer1.css';

export const Layer1Explore = () => {
	const { user } = useUser();
	const ExploreUrl = user ? '/store' : '/signIn';

	return (
		<div className="exploreButtonBack">
			<Link className={['exploreButton'].join(' ')} to={ExploreUrl} type="button">
				{' '}
				Explore{' '}
			</Link>{' '}
			{'  '}{' '}
		</div>
	);
};
class Layer1Left extends React.Component {
	constructor(props) {
		super(props);
		this.state = {};
	}

	render() {
		return (
			<div className="home-layer1-left-box">
				<div className="welcome-container">
					<h2 className="welcome-header"> Click to Earn Coins and Collect Art! </h2>{' '}
					<span className="welcome-desc"> Welcome to the Art Market with the Most Clicks! </span>
					<Layer1Explore />
				</div>
			</div>
		);
	}
}

// class Layer1Right extends React.Component
const Layer1Right = ({ data }) => {
	const collections = data;
	const [imgurl, setImgurl] = useState([]);
	const [username, setName] = useState([]);
	const [artname, setArtname] = useState([]);
	const navigate = useNavigate();
	const { user } = useUser();
	const ExploreUrl = user ? '/store' : '/signIn';

	useEffect(() => {
		initial(collections);
	}, [collections]);

	const initial = (cols) => {
		if (cols && cols.length) {
			const col0 = cols[0];
			setImgurl(client.getImg(col0.img_url));
			setArtname(col0.display_name);
			setName(col0.username);
		} else {
			setImgurl(layer1RightIm2);
			setArtname('this is art name');
			setName('this is the author name');
		}
	};

	const imageClick = () => {
		// alert('WILL GO TO THE DETAILED PAGE!'); // eslint-disable-line no-alert
		navigate(ExploreUrl);
	};

	return (
		// <div className="flexboxBack">
		<div
			className={[
				css`
					${'' /* background-color:red; */}
					@media only screen and (max-width: 700px) {
						display: none;
						transition: all 0.2s ease;
					}
				`,
				'home-layer1-right-box',
			].join(' ')}
		>
			<div className="collection-card-container">
				<img src={imgurl} onClick={() => imageClick(imgurl)} alt="123" /* eslint-disable-line*/ />
				<div className="flexbox-right-footer">
					<div className="flexbox-right-footer-left">
						<img src={imgurl} alt="123" />
					</div>{' '}
					<div className="flexbox-right-footer-right">
						<h2 className="flexbox-right-footer-right-item above"> {artname} </h2>{' '}
						<div className="flexbox-right-footer-right-item"> {username} </div>{' '}
					</div>{' '}
				</div>{' '}
			</div>
		</div>
	);
};

const Layer1 = ({ data }) => (
	<div className="home-layer1-main-box">
		<Layer1Left />
		<Layer1Right data={data} />{' '}
	</div>
);

export default Layer1;
