import React, { useState } from 'react';
import { IconContext } from 'react-icons';
import { BiMenu, BiX } from 'react-icons/bi';
import { useUser } from '../../context/userContext';
import Dropdown from './Dropdown';
import { Nav, NavbarContainer, NavLogo, NavIcon, MenuIcon, Menu, MenuItem, MenuLink } from './styles';

const UnauthenticatedMenu = ({ closeMenu }) => (
	<>
		{/* <MenuItem>
			<MenuLink onClick={closeMenu} to="/explore">
				Explore{' '}
			</MenuLink>{' '}
		</MenuItem>{' '} */}
		<MenuItem>
			<MenuLink onClick={closeMenu} to="/signIn">
				Sign In{' '}
			</MenuLink>{' '}
		</MenuItem>{' '}
	</>
);

const AuthenticatedMenu = ({ closeMenu }) => (
	<>
		<MenuItem>
			<MenuLink onClick={closeMenu} to="/mycollection">
				Collection
			</MenuLink>
		</MenuItem>
		<MenuItem>
			<MenuLink onClick={closeMenu} to="/clicker">
				Clicker{' '}
			</MenuLink>{' '}
		</MenuItem>{' '}
		<MenuItem>
			<MenuLink onClick={closeMenu} to="/store">
				Store{' '}
			</MenuLink>{' '}
		</MenuItem>{' '}
		<MenuItem>
			<MenuLink onClick={closeMenu} to="/trade">
				Trade
			</MenuLink>
		</MenuItem>
		<MenuItem>
			<Dropdown />
		</MenuItem>{' '}
	</>
);

const Navbar = () => {
	const { user } = useUser();

	const [click, setClick] = useState(false);
	const handleClick = () => setClick(!click);
	const closeMenu = () => setClick(false);

	return (
		<div>
			<IconContext.Provider value={{ color: 'rgba(4, 17, 29, 0.75)' }}>
				<Nav>
					<NavbarContainer>
						<NavLogo to="/">
							<NavIcon />
							<h2> Clickmania</h2>{' '}
						</NavLogo>{' '}
						<MenuIcon onClick={handleClick}> {click ? <BiX /> : <BiMenu />} </MenuIcon>
						<Menu onClick={handleClick} click={click}>
							{' '}
							{user ? (
								<AuthenticatedMenu closeMenu={closeMenu} />
							) : (
								<UnauthenticatedMenu closeMenu={closeMenu} />
							)}{' '}
						</Menu>{' '}
					</NavbarContainer>{' '}
				</Nav>{' '}
			</IconContext.Provider>{' '}
		</div>
	);
};

export default Navbar;
