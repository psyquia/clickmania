import styled from '@emotion/styled';
import { Link } from 'react-router-dom';
import { IoMdColorPalette } from 'react-icons/io';

export const Nav = styled.nav`
	font-size: 18px;
	position: sticky;
	top: 0;
	z-index: 999;
	height: 80px;
	box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.15);
	display: flex;
	justify-content: center;
	align-items: center;
`;

export const NavbarContainer = styled.div`
	display: flex;
	justify-content: space-between;
	align-items: center;
	height: 80px;
	margin: 0 auto;
	padding: 0 50px;
	width: 100%;

	@media (max-width: 400px) {
		padding: 0 10px;
	}
	@media (max-width: 991px) {
		padding: 0 30px;
	}

	@media (min-width: 1500px) {
		max-width: 1500px;
	}

	@media (min-width: 1800px) {
		max-width: 1800px;
		padding: 0 30px;
	}
`;

export const NavLogo = styled(Link)`
	color: rgba(4, 17, 29, 0.75);
	cursor: pointer;
	display: flex;
	align-items: center;
	text-decoration: none;
	font-size: 2.5rem;
	font-weight: 800;
	transition: all 0.5s ease;
	&:hover {
		transform: scale(1.08);
	}
`;

export const NavIcon = styled(IoMdColorPalette)`
	margin-right: 0.8rem;
	transition: all 0.5s ease;

	&:hover {
		transform: scale(2);
	}
`;

export const MenuIcon = styled.div`
	display: none;

	@media (max-width: 750px) {
		display: block;
		position: absolute;
		top: 0;
		right: 0;
		transform: translate(-50%, 20%);
		font-size: 4rem;
		cursor: pointer;
	}
`;

export const Menu = styled.ul`
	display: flex;
	align-items: center;
	text-align: center;

	@media only screen and (max-width: 750px) {
		display: flex;
		flex-direction: column;
		width: 100%;
		height: 100vh;
		position: absolute;
		top: 80px;
		left: ${({ click }) => (click ? '0' : '-100%')};
		background-color: rgba(255, 255, 255, 0.9);
		transition: all 0.5s ease;
	}
`;

export const MenuItem = styled.li`
	list-style: none;
	height: 80px;

	@media only screen and (max-width: 750px) {
		width: 100%;
		&:hover {
			border: none;
		}
	}
`;

export const MenuLink = styled(Link)`
	text-decoration: none;
	font-weight: bold;
	font-size: 2rem;
	color: rgba(4, 17, 29, 0.75);
	display: flex;
	justify-content: space-between;
	align-items: center;
	padding: 1rem 2rem;
	height: 100%;
	transition: all 0.2s ease;

	&:hover {
		color: #000;
		transform: traslateY(-3rem);
	}
	&:active {
		transform: traslateY(3rem);
		color: #000;
	}

	@media only screen and (max-width: 750px) {
		display: block;
		padding: 3rem;
		text-align: center;
		transition: all 0.2s ease;
	}
`;

export const ButtonLink = styled.div`
	text-decoration: none;
	font-weight: bold;
	font-size: 2rem;
	color: rgba(4, 17, 29, 0.75);
	display: flex;
	justify-content: space-between;
	align-items: center;
	padding: 1rem 2rem;
	height: 100%;
	transition: all 0.2s ease;

	&:hover {
		color: #000;
		transform: traslateY(-3rem);
		cursor: pointer;
	}
	&:active {
		transform: traslateY(3rem);
		color: #000;
	}

	@media only screen and (max-width: 750px) {
		display: block;
		padding: 3rem;
		text-align: center;
		transition: all 0.2s ease;
	}
`;
